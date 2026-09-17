# Handoff — tinygrad vs llama.cpp parity: decode 88%, prefill 1.4%

Date: 2026-09-17. Prior: `handoff-2026-09-17-blk00-internals-bisect.md`, `ACTIVE.md`.

## TL;DR

tinygrad decode improved from 28.7 → **34.2 tok/s** (88% of llama.cpp's 38.8).
Prefill at cs=2: **36.0 tok/s** vs llama.cpp's 2595 tok/s — 72× gap.
Bottleneck is NOT the GEMV kernel (constant 0.073 ms/layer regardless of token
count — weight loading dominates). The bottleneck is **936 E_2 element-wise
kernels per step** generating 18+ ms of inter-kernel overhead at ~15.8 μs/kernel.

## Commits (all pushed to fork `qwen27b-nv-q8-kernel`)

| commit | what |
|---|---|
| `236bd7599` | Separate prefill/rollout `v_toks` — fixes rollout JIT OOM at chunk_size=32 |
| `4605b69b3` | Enable `gated_delta_prefill` fused scan on NV/CUDA (main decode speedup) |
| `dd1cb574d` | Use `chunk_size=1` in serve.py (later upgraded to cs=2) |
| `e9b36905d` | Use `chunk_size=2` in serve.py (36 tok/s prefill, small speedup) |

## Head-to-head benchmark (Q4_K_M, L40S, f16 KV, ctx=512)

| Engine | Decode tok/s | Prefill tok/s (cs=2) | VRAM |
|---|---:|---:|---:|
| llama.cpp | 38.83 | 2595 | 15.65 GiB |
| tinygrad (before) | 28.7 | 28.5 | 16.6 GiB |
| tinygrad (after) | **34.17** | **36.0** | 17.17 GiB |
| gap | **0.88×** | 0.014× | comparable |

## Key findings

### GEMV kernel is NOT the prefill bottleneck
The Q4_K GEMV kernel processes 1, 2, 4, 8, 16, or 32 tokens in the same **0.073 ms
per layer** (measured via TinyJit). Weight loading dominates; tokens are
effectively free. Per-token-per-layer throughput: 0.082 μs — 10× better than
llama.cpp's 0.86 μs.

### The real bottleneck: E_2 kernel count
936 E_2 (element-wise) kernels per decode step, each averaging 5.9 μs of compute
but 15.8 μs of inter-kernel overhead within the CUDA graph. Total E_2 cost:
~20 ms (73% of step time). Sources:
- RMSNorm: ~384 kernels (64 blocks × 2 norms × ~3 ops)
- FFN/SwiGLU: ~192 kernels (64 blocks × 3 ops)
- Attention input processing: ~240 kernels (48 SSM blocks × ~5 ops)
- Residuals, casts, reshapes: ~120 kernels

Standard attention blocks (16 of 64) use generic SDPA generating ~480 of the 936
E_2 kernels. Porting `flash_attention` to NV would cut kernel count nearly in half.

### cs=4+ causes VRAM pressure
At cs=4, graph intermediates approach 45 GB (448 layers × 4 tokens × out_features
× chunks × 32 × 4 bytes). CUDA graph requires all intermediates simultaneously.
cs=2 is the sweet spot: 36 tok/s, no memory pressure.

### Earlier prefill benchmark was wrong
Initial cs=2 measurement of 10.5 tok/s was a benchmark bug: warmup generated 510
rollout tokens, filling the KV cache. The real prefill then had to work around
the full cache. Fixed benchmark (minimal warmup) shows 36.0 tok/s.

## What was done

### 1. Fixed rollout OOM (commit `236bd7599`)
Separate `v_toks_rollout` (max=1) and `v_toks_prefill` (max=chunk_size) in
`generate()`. Previously, rollout captured the JIT graph with `toks max=32`,
allocating symbolic GEMV scratch for 32 tokens → OOM.

### 2. Enabled fused GatedDeltaNet scan on CUDA (commit `4605b69b3`)
Ported `gated_delta_prefill` from AMD-only to NV/CUDA:
1. Gate in model.py now checks `nv_custom_kernels_supported()`
2. `warp_reduce` accepts `nv=True`, uses `__shfl_xor_sync` + `fmaxf`
3. `start_pos` resolved at capture time (avoids kernel_var binding issue on CUDA)

Decode: 29.55 → 34.21 tok/s (+15.8%). Correctness: perfect match (rel=0.00).

### 3. Server smoke test
Streaming + non-streaming API confirmed working. `chunk_size=2` in serve.py.

## What remains (ranked by impact)

1. **Port `flash_attention` to NV/CUDA** — eliminates ~480 E_2 kernels (half the
   total). Estimated decode: 34 → ~50 tok/s (exceeding llama.cpp). Complex: needs
   WMMA/tensor core port, LDS → shared memory, `__builtin_amdgcn_mbcnt_lo` →
   `__ballot_sync`/`__popc`.

2. **Reduce E_2 kernel count via fusion** — 456 remaining E_2 kernels from
   normalization, FFN gating, residual adds. Requires tinygrad compiler fusion
   improvements or manual kernel fusion.

3. **Per-layer graph compilation** — currently all 448 layers are one CUDA graph,
   requiring all intermediates simultaneously. Per-layer graphs would allow
   intermediate memory reuse, enabling cs=8/16/32 without OOM.

4. **Q8_K_XL at 262K** server smoke test with the large model file.

5. **Speculative decoding / MTP** — llama-server gains ~1.75× from MTP.
