# Handoff — chunked-prefill correctness, current HEAD, consolidated 2026-09-22

Supersedes: `docs/HANDOFF-2026-09-22.md`, `docs/p0-p7-status-2026-09-22.md`
(both written by an earlier agent on stale assumptions — see §7).
Canonical partners: `handoff-2026-09-19-tdep-probe.md`, `ACTIVE.md`, `progress.md`.

## 1. Question and answer

**Q: is chunked prefill (cs>=2) safe to re-enable on current HEAD?**
**A: no.** Full-model gate on node-lair H100, same 24-token prompt, same model
(`Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`), HEAD `12e1f301d`:

| run | script | result |
|---|---|---|
| cs=1 | `compare_logits.py 1` | argmax **271**, top-1 34.34, gap 13.7 ← correct |
| cs=2 | `compare_logits.py 2` | argmax **220**, top-1 25.03, gap 0.12 ← wrong, near-flat |
| cs=4 | `compare_logits.py 4` | argmax **220**, top-1 24.73 ← wrong, near-flat |

`serve.py` stays at `chunk_size=1`. This reproduces the 09-17 finding on current HEAD.

## 2. What ran today (node-lair H100, 2026-09-22)

- `bisect_blocks.py` cs=1 vs cs=2 (24-token prompt): **diverges at blk00
  (first block), mean rel 7.09e-01** — `a_mean=+1.71e-03` vs `b_mean=+5.87e-03`.
  (This path stops at first DIVERGE, so it does not separate embedding-noise
  from blk00-amplification on its own; the 09-17 `bisect_blk00_internals.py`
  result does: embedding first at rel 3.4e-04, scan/pad/state verified correct.)
- Item 7 (full blk0 `_attention`): crashes in `Tensor.finalize_after`
  (`movement.py:211`, `self.ndim=3 != len(arg)=4`) even with the script-side
  workarounds (4-arg-shrink fix, realize-in-numpy, realize-before-zero-state).
  This is a tinygrad graph-internal issue on the 4-D state tensors, not our code.
- **Flash-attention verification (task 2, step 1)**: JIT=0 DEBUG=2 census of
  2 decode steps on node-lair H100 = **2397 kernels/step** (matches 09-17 exactly).
  `flash_decode_partial` fires **16/step** (one per attention block) on CUDA;
  `gated_delta_prefill` fires 48/step (48 SSM blocks). `E_*` elementwise count is
  now **436/step** - down from ~936 pre-FA-port, confirming `0f7bd750d` eliminated
  ~500 kernels as predicted. Decode measured **37.6 tok/s (26.6 ms/tok)** H100.
  Remaining decode headroom: fuse ~436 `E_*` + ~900 generic `r_*` kernels.
  Full census: node-lair `/tmp/fadump_raw.txt` (ANSI-strip; line fmt `*** CUDA <n> <name>`).

## 3. Interpretation

- The probe uses **concrete** shapes; `generate()` uses **symbolic** `toks`
  variables bound at JIT capture. Embedding is exact with concrete shapes but
  JIT-dependent in the symbolic path (09-17 measured rel 3.4e-04 there).
  Item 7's crash therefore does not contradict the full-model gate — the gate
  is the authoritative answer, and it says cs>=2 is wrong.
- The embedding `patch_embedding_gather.py` was written 09-17 but
  **reverted after failing to change the divergence** (see
  `handoff-2026-09-17-chunked-prefill-wrong.md` §"What to try next" item 1).
  It is not an untested silver bullet — it was tested and did not move the needle.
- Per-block bisect from 09-17 (`handoff-2026-09-17-blk00-internals-bisect.md`):
  embedding diverged first (rel 3.4e-04), blk00 amplified to rel 2.58;
  scan/pad/state logic verified correct; padded step exact no-op.

## 4. P0–P7 ledger (evidence, not vibes)

| P | status | evidence |
|---|---|---|
| P0 profile | COMPLETE | `progress.md` session 3; kernel census 2282/step, Q6_K 51.4 ms found |
| P0 T-probe items 1–6 | COMPLETE | this session: bit-identical ×5, scan 9.39e-04 |
| P0 item 7 | BLOCKED | tinygrad `finalize_after` shrink bug on 4-D state; script-side fixes exhausted |
| P0 full-model gate | COMPLETE (negative) | cs=1: 271/34.34; cs=2: 220/25.03; cs=4: 220/24.73 |
| P1 DeltaNet | NOT STARTED | needs settled prefill story, still open |
| P2 chunked prefill | DECIDED: stays off | gate fails; `serve.py` cs=1 |
| P3 Q4_K micro-opts | COMPLETE | `kernels/nv_q4k.py`, 12.87 tok/s 5.03× |
| P4 Q6_K kernel | COMPLETE | `kernels/nv_q6k.py`, sweep 7/7 |
| P5 BEAM_CACHE | UNVERIFIED | training never completed |
| P6 MTP | DEFERRED | low ROI |
| P7 hygiene | PARTIAL | `tinygrad/llm/gguf_q3k.py` framework; Q3_K route + logit A/B + llama cross-check open |

## 5. Next steps (ranked) — updated 2026-09-22 end-of-session

DONE this session (do not redo):
- Bisect landed: diverges at blk00, mean rel 7.09e-01 (cs=1 1.71e-03 vs cs=2 5.87e-03).
- P0 closed: cs>=2 unsafe on current HEAD (gate + bisect + 09-17 internals agree).
- `serve.py` verified pinned at `chunk_size=1` (node tree reset to fork HEAD `12e1f301d`).
- Server smoke: `\n\nParis` on France prompt, port 8000, H100. Server stopped.
- **Task 2 step 1 (FA verification): DONE.** `flash_decode_partial` fires 16/step
  on CUDA; `gated_delta_prefill` 48/step; `E_*` down to 436/step (was ~936);
  decode 37.6 tok/s (26.6 ms/tok) H100 vs llama.cpp 80.9.

REMAINING (in order):

1. **Fuse ~436 `E_*` + ~900 generic `r_*` kernels/step** (decode headroom:
   37.6 -> target ~50+ tok/s). Top families: `r_16_320` (129), `E_40_32_4` (129),
   `r_136_32_4_5` (128), `E_320_32_3` (96), `r_3_16_5` (96), `r_16_16_8` (96)
   - RMSNorm/FFN-gate/residual chains. Engine/codegen work: custom kernels are
   fusion barriers (measured worse twice). Target the biggest three families first.
2. **P7 hygiene (mechanical, unblocks Unsloth UD files):** Q3_K(11)/Q8_K(15)/
   IQ4_NL loader in `tinygrad/llm/gguf.py` `_GGML_QUANT` + `amd.py` routing;
   complete `tinygrad/llm/gguf_q3k.py` dequant (ggml `dequantize_row_q3_K` bit
   logic; framework + block layout already there). Verify with
   `check_q6k_loader.py` pattern on real UD-Q4_K_M bytes.
3. **logit-diff A/B (not argmax-only)** for every future kernel change:
   reuse `compare_logits.py`, print full top-5 + max|d| rel, not just argmax.
4. **llama.cpp cross-check on the Q4_K_M file** (`/N/slate/demistry/llama.cpp/
   build/bin/llama-bench`) - closes the parity table on the same file.
5. **Q8_K_XL 262K smoke re-check** on current HEAD (was verified 09-06,
   37.8 GiB at 262144 ctx; re-check after FA/chunk changes).
6. **P2 stays off.** Prefill 72x behind llama.cpp; structural causes documented
   (`handoff-2026-09-17-root-cause-analysis.md`): no tensor-core matmul
   (6.4 TFLOPS of ~990 peak) + per-node latency floor.

NON-GOALS (do not re-litigate):
- item 7 `finalize_after` crash: tinygrad-internal, script-side fixes exhausted
- gather embedding patch: tried 09-17, reverted, did not move divergence
- cs>=2: gate failed three independent ways (logits, per-block bisect, internals)

## 6. Environment notes (node-lair H100, verified today)

- Model: `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
- tinygrad: `~/tinygrad-src`, branch `qwen27b-nv-q8-kernel`, HEAD `12e1f301d`
- Env that works: `PYTHONPATH=~/tinygrad-src DEV=CUDA`, **no `CUDA_PATH`**.
  Setting `CUDA_PATH=/usr/lib/x86_64-linux-gnu` breaks NVRTC includes
  (`compiler_cuda.py:51` uses only `-I$CUDA_PATH/include`); unset falls back
  to `-I/usr/local/cuda/include -I/usr/include -I/opt/cuda/include`, which works.
- Scripts on node `/tmp`: `tdep_probe_fixed.py` (local `tdep_probe.py` +
  model-path sed), `compare_logits.py`, `isolate_chunks.py` (same treatment).
- Each `compare_logits.py` run: one model load (~60 s) + JIT, ~2 min total.
- `bisect_blocks.py` cs=1 vs cs=2: two loads + JIT + 65-block shrink stats,
  still running after ~10 min at time of writing.

## 7. Corrections to earlier 2026-09-22 agent docs

- `p0-p7-status-2026-09-22.md` and `HANDOFF-2026-09-22.md` claim GPU execution
  was "BLOCKED on quartz ... libcuda.so load fails" and "NVRTC cannot find
  cuda_fp16.h". Both were wrong diagnoses: the failure was a self-inflicted
  `CUDA_PATH=/usr/lib/x86_64-linux-gnu`, which narrows NVRTC includes to a
  nonexistent dir. Unset `CUDA_PATH` compiles and runs fine on node-lair H100.
  Treat those two files as superseded; `tdep_probe.py` (tracked, untracked in
  git) is the only script change from that session worth keeping.
- Uncommitted local state: `M docs/ACTIVE.md`, `?? docs/HANDOFF-2026-09-22.md`,
  `?? docs/handoff-2026-09-19-tdep-probe.md`,
  `?? docs/p0-p7-status-2026-09-22.md`, `?? scripts/tdep_probe.py`.
