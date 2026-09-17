# Handoff — tinygrad vs llama.cpp parity: decode 88%, prefill gap documented

Date: 2026-09-17. Prior: `handoff-2026-09-17-blk00-internals-bisect.md`, `ACTIVE.md`.

## TL;DR

tinygrad decode improved from 28.7 → **34.2 tok/s** (88% of llama.cpp's 38.8).
Prefill remains at 34.2 tok/s (cs=1) vs llama.cpp's 2595 tok/s — 76× gap,
requires quantized GEMM kernels (future work).

## Commits (all pushed to fork `qwen27b-nv-q8-kernel`)

| commit | what |
|---|---|
| `236bd7599` | Separate prefill/rollout `v_toks` — fixes rollout JIT OOM at chunk_size=32 |
| `4605b69b3` | Enable `gated_delta_prefill` fused scan on NV/CUDA (main decode speedup) |
| `dd1cb574d` | Use `chunk_size=1` in serve.py (avoids prefill JIT OOM) |

## Head-to-head benchmark (Q4_K_M, L40S, f16 KV, ctx=512)

| Engine | Decode tok/s | Prefill tok/s | VRAM |
|---|---:|---:|---:|
| llama.cpp | 38.83 | 2595 | 15.65 GiB |
| tinygrad (before) | 28.7 | 28.5 | 16.6 GiB |
| tinygrad (after) | **34.17** | 34.17 | 17.17 GiB |
| gap | **0.88×** | 0.013× | comparable |

## What was done

### 1. Fixed rollout OOM (commit `236bd7599`)
`generate()` used a single `v_toks` with `max=chunk_size` for both prefill and
rollout. The rollout path (1 token) still captured the JIT graph with
`toks max=32`, allocating symbolic GEMV scratch for 32 tokens → OOM at 43.75 GB.

Fix: separate `v_toks_rollout` (max=1) and `v_toks_prefill` (max=chunk_size).

### 2. Enabled fused GatedDeltaNet scan on CUDA (commit `4605b69b3`)
The `gated_delta_prefill` fused kernel was AMD-only. Three changes:

1. **model.py**: Gate now checks `nv_custom_kernels_supported()` in addition to
   `amd_custom_kernels_supported()`.
2. **amd.py `warp_reduce`**: Accepts `nv=True` flag, uses `__shfl_xor_sync` +
   `fmaxf` (via CUSTOMI) on NV instead of `__builtin_amdgcn_ds_swizzle`.
3. **amd.py `gated_delta_prefill`**: Resolves `start_pos` at capture time
   (prefill=0 → reset state, rollout>0 → keep state) instead of using
   `kernel_var` binding that doesn't propagate on CUDA.

**Impact**: Eliminates 936 `E_2` element-wise kernels per decode step (52% of
GPU time). Decode: 29.55 → 34.21 tok/s (+15.8%).

### 3. Server smoke test
Server starts, serves streaming responses via `/v1/chat/completions`. Fixed
prefill OOM by setting `chunk_size=1` in serve.py. Non-streaming mode returns
proper JSON. Streaming works via SSE.

### 4. Prefill analysis
Prefill at cs>1 is **slower** than cs=1 because the custom GEMV kernels process
each token independently (no weight sharing across tokens). The generic matmul
path (dequantize + GEMM) is even slower. Closing the prefill gap requires
**quantized GEMM kernels** that batch multiple tokens against packed weights —
a separate kernel project.

## What remains (ranked by impact)

1. **Quantized GEMM kernels for prefill** (76× gap). Write batched Q4_K/Q6_K
   GEMM kernels that process multiple tokens against packed weights in a single
   matrix multiply. This is how llama.cpp achieves 2600 tok/s prefill.

2. **Decode: close remaining 12% gap** (34.2 vs 38.8). Potential:
   - Flash attention for standard attention blocks (currently AMD-only gate at
     model.py:219)
   - Further GEMV micro-optimization (`__ldcs` streaming loads for Q4_K)

3. **Q8_K_XL at 262K context**: server smoke test with the large model file.
   Currently tested with Q4_K_M at ctx=512.

4. **Speculative decoding / MTP**: llama-server gains ~1.75× from MTP. Not
   measured in llama-bench (38.83 is without MTP).
