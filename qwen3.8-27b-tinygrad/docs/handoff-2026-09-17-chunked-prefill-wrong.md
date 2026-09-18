# Handoff 2026-09-17 (late): chunked prefill is numerically WRONG — server now uses cs=1

**Read this before running anything with `chunk_size >= 2`.**

## The finding

Chunked prefill produces different — and semantically wrong — output than token-by-token
prefill. Same model, same prompt, greedy (temp=0), H100:

| chunk_size | argmax token | top-1 logit | margin over top-2 | decoded |
|---:|---:|---:|---:|---|
| 1 | 271 | 34.34 | 13.7 | `"\n\nParis"` ← correct |
| 2 | 220 | 25.03 | 0.12 | whitespace |
| 4 | 220 | 24.73 | 2.1 | whitespace |

Prompt: the model's own chat template for *"What is the capital of France? Answer in one word."*
(24 tokens).

**Control:** cs=1 vs cs=1 → `rel = 0.00e+00` (bit-identical). The harness is sound; the
divergence is real and deterministic.

## Where it comes from

`scripts/localize_divergence.py` compares token 0's activations between cs=1 (1-token chunk)
and cs=4 (4-token chunk) for the same prompt and position:

| stage | max\|x\| | max\|d\| | rel |
|---|---:|---:|---:|
| embedding output | 4.26e-01 | 1.16e-04 | 2.7e-04 |
| blk00 (GatedDeltaNet) output | 1.75e+01 | 4.53e+01 | **2.58** |

Two facts:

1. **The embedding is T-dependent.** `nn._embedding_fwd` is a one-hot reduction over the whole
   vocabulary (`(arange == idx).where(weight, 0).sum(-2)`), so its reduction order follows the
   kernel shape, which follows the token count. Different `toks` → different fp32 rounding →
   ~1e-4.
2. **blk00 amplifies it ~10⁴×.** A 1.16e-4 input difference becomes a 2.58 relative difference
   in the *first* SSM block. Likely the `normalize(q,k)` (eps=1e-12 for KDA) and/or the delta-rule
   recurrence; not yet isolated further.

Multi-chunk runs additionally accumulate this per chunk, which is why cs=2 and cs=4 both land on
token 220 with a near-flat distribution (the model has lost the prompt).

## Status

- **`serve.py` now uses `chunk_size=1`** (commit `12e1f301d`, pushed to
  `fork/qwen27b-nv-q8-kernel`). Cost: ~5% prefill throughput (34.4 vs 36.0 tok/s on L40S).
  Correctness wins.
- The same commit range also carries `3c92cf1a4` (flash_attention gated to `resolve(T == 1)`,
  without which any chunked prefill crashes on `chunk_size must be a multiple of 32`).
- **This bug is pre-existing, not introduced by the FA work.** Before FA was enabled on NV, the
  NV attention path was already the standard one for all T — same code path cs≥2 uses now. It was
  only *unreachable* while FA was unconditionally enabled (it crashed), and before that its output
  was never checked for correctness — the earlier "server smoke test works" only proved the API
  responded.

## What to try next (in order)

1. **Make the embedding exact.** Replace the one-hot reduction with a gather. I patched
   `_embedding_fwd` to `return weight[idx]` and re-measured: **the 1.16e-4 difference did not
   change**, so either the gather is not lowered to a plain copy or the difference originates
   downstream. Reverted (unverified changes don't ship). Verify with
   `scripts/localize_divergence.py` — `emb` must read `max|d| = 0.0` before moving on.
2. **Then attack the amplification.** Even with an exact embedding, check whether other kernels
   are T-dependent (attention reductions, RMSNorm, the output projection). The SSM block turning
   1e-4 into 2.6 means *any* upstream noise is fatal, so bit-exactness across T is the bar.
3. **Regression gate.** `scripts/isolate_chunks.py` + `scripts/compare_logits.py` are the test:
   cs=1,2,4 must produce the same argmax and a matching top-5.

## Reproduction

```bash
ssh quartz 'salloc --account=r00117 --partition=h100-single --gres=gpu:1 --mem=128G -t 4:00:00 bash'
# then ssh to the granted node (config alias: qgpu)
cd /N/slate/demistry/tinygrad-src
CUDA_PATH=/N/soft/rhel8/cuda/12.6 DEV=CUDA python3 /tmp/compare_logits.py 2
CUDA_PATH=/N/soft/rhel8/cuda/12.6 DEV=CUDA python3 /tmp/localize_divergence.py
```
Each script does one full model load (~60 s) + JIT compile.

## Performance summary (unchanged by this finding)

| GPU | engine | decode | prefill |
|---|---|---:|---:|
| L40S 46 GB | llama.cpp | 38.83 tok/s | 2595 tok/s |
| L40S 46 GB | tinygrad | 34.2 tok/s | 36.0 tok/s (cs=2, **incorrect**) / 34.4 (cs=1) |
| H100 80 GB | llama.cpp | 80.88 tok/s | 2424 tok/s |
| H100 80 GB | tinygrad | 42.11 tok/s | 38.7 tok/s (cs=2, **incorrect**) |

Root causes for the throughput gaps are in
`handoff-2026-09-17-root-cause-analysis.md` (no working tensor-core matmul on CUDA: 6.4 TFLOPS vs
~990 peak; per-node latency floor).
