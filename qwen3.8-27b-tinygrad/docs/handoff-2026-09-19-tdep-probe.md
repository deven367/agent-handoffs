# Handoff 2026-09-19: T-dependence probe — chunked-prefill correctness

**Goal:** determine on current HEAD (`12e1f301d`) exactly what is T-dependent,
to decide whether chunked prefill (cs>=2) can be safely re-enabled.
Motivation: 09-17 late handoff found cs>=2 produces wrong tokens; serve.py at
chunk_size=1 (34.4 tok/s prefill vs 36.0 at cs=2 which is wrong).

## Environment (quartz / H100)

- Node: **g37** via `ssh node-quartz` (quartz job 10523323, partition
  `h100-debu`, H100 80GB). **Job expired — re-request:**
  ```bash
  ssh quartz "salloc --account=r00117 --partition=h100-single --gres=gpu:1 --mem=128G -t 4:00:00 bash"
  ssh node-quartz
  ```
  (local ssh config: `node-quartz` -> g37, `quartz` -> login node; `qgpu` is stale -> g73)
- tinygrad: `/N/slate/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`,
  HEAD `12e1f301d`, clean tree.
- Model: `/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
- Env: `export CUDA_PATH=/N/soft/rhel8/cuda/12.6 DEV=CUDA`
- Model dims (Qwen3.8-27B, GDN block 0): Hk=16, Hv=48, Dk=128, Vd=128, inner=6144

## Probe: `scripts/tdep_probe.py` (copy to /tmp on node)

Compares T=1 vs T=4, token 0, identical input, on each GDN-block primitive
(realized values). Inputs = slices of the embedding weight (deterministic,
already on CUDA).

```bash
python3 /tmp/tdep_probe.py
```

### Results (g37, 2026-09-19)

| # | stage | T=1 vs T=4 token 0 |
|---|-------|--------------------|
| 1 | embedding (one-hot reduction over vocab 248320) | **rel 0.00e+00 — bit-identical** |
| 2 | attn_qkv GEMM (Q4_K) | **rel 0.00e+00 — bit-identical** |
| 3 | ssm_out GEMM | **rel 0.00e+00 — bit-identical** |
| 4 | qk dot (sum over Dk=128) | **rel 0.00e+00 — bit-identical** |
| 5 | normalize(dim=-1, eps=1e-12) [fp16] | **rel 0.00e+00 — bit-identical** |
| 6 | fused scan, token 0 (zero state, reset branch) | max\|d\|=9.05e-08, **rel 9.39e-04** |
| 7 | full blk0 `_attention` (identical input) | **CRASHED** (see below) |

## Implications

1. The 09-17 "JIT-dependent embedding" (rel 3.4e-04 in the symbolic cs=1-vs-cs=2
   bisect) does **not** reproduce on current HEAD with concrete shapes: embedding,
   both GEMMs, qk dot, and the eps=1e-12 normalize are all bit-identical for T=1
   vs T=4.
2. The scan (item 6) is nearly T-independent with **zero initial state** (~1e-3
   rel, outputs ~1e-4 in magnitude — consistent with rounding in `kq =
   (q*k).sum(-1)` / delta·kq terms). Non-zero state and 48 SSM blocks × 64 layers
   is where the 09-17 blk00 output rel 2.58 likely comes from.
3. Residual candidates NOT covered by the probe:
   - **T_pad path** (top suspect): `conv_window` buffer size
     `conv_kernel-1 + T_pad`, `out_gate`/`beta`/`log_alpha`
     `pad_to((B, T_pad, ...))`, where `T_pad = x.max_shape[1]` (1 vs 4).
   - Full 64-block accumulation (16 attention blocks use SDPA — untested).
   - Scan with carried-over (non-zero) state.

## What to run next (in order)

1. **Finish item 7.** The probe crashed on a `shrink` in
   `tinygrad/mixin/movement.py` (3D tensor, 4-element slice). `o4full`/`o1full`
   shapes are now printed to stdout. If `o4full` is 4D, adjust the slice in
   `diff(o1full, o4full[:, :1])`. (The `b._init_state(x)` call before zeroing
   is required — state attrs are lazily created, not module-level.)
2. **If item 7 rel ~ 0:** run the full-model gate:
   ```bash
   python3 /tmp/compare_logits.py 1
   python3 /tmp/compare_logits.py 4     # (and 2)
   ```
   Same argmax + top-5 logits → chunked prefill is safe → set
   `chunk_size=2` in `serve.py` (36.0 tok/s, +5% over cs=1).
3. **If item 7 diverges:** run `scripts/bisect_blk00_internals.py` (cs=1 vs
   cs=2, token 0) and find the first divergent stage; the T_pad conv window is
   the prime suspect.
4. Regression gate for any cs re-enable: `compare_logits.py` + `isolate_chunks.py`
   — cs=1/2/4 same argmax, matching top-5.

## Gotchas learned (probe debugging)

- **`Tensor.to(device)` is a no-op for CPU tensors** —
  `tinygrad/tensor.py:543`: `if self.uop.device is None: return self`. Docstring
  says "Moves the tensor to the given device." Build inputs directly on CUDA
  (slices of a model weight) or fix `to`.
- **`Tensor([0.0])`** builds a model-parameter BUFFER (arg=0), not a constant.
  Scan `start_pos` must be `Tensor(UOp.variable("sp", 0, max_ctx-1).bind(0))`
  (a bound variable); `Tensor(0)` is a CONST and `unbind()` rejects it.
- **Scan kernel writes back into the input `state` buffer** — use a fresh zero
  state per call.
- **Scan expects fp32 inputs** (model casts `.float()` before calling); half
  inputs give a `half4`/`float4` type error in generated CUDA.
- **Scan input layout:** q/k are `(B, Hv, T, Dk)` — the model repeats q/k to
  `num_v_heads` (`.repeat(1, 1, Hv//Hk, 1)`) before the call; v/beta/alpha are
  `(B, Hv, T, ...)`; state `(B, Hv, Vd, Dk)` fp32.
- **4-D slice:** token 0 of a `(B, H, T, K)` tensor is `[:, :, :1]`, not `[:, :1]`.
- **CPU-routed ops crash on this machine** — gcc rejects
  `--target=x86_64-none-unknown-elf`. Every tensor must be on CUDA.
- GDN block state attrs (`conv_state`, `recurrent_state`) are created lazily in
  `GatedDeltaNetBlock._init_state(x)` (model.py:381-384); call it before
  calling `_attention` directly.

## State after this session

- serve.py: chunk_size=1 (unchanged, correct).
- tinygrad tree: clean, no edits made (probe was read-only).
- Docs: this file + updated ACTIVE.md are the new canonical state.
- L40S (node-lair) path untouched: `/u/demistry/tinygrad-src`, same branch.
