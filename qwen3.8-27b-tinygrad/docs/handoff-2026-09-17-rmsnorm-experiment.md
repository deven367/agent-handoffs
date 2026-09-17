# Handoff — RMSNorm fusion experiment + final bottleneck analysis

Date: 2026-09-17. Prior: `handoff-2026-09-17-parity-benchmark.md`.

## TL;DR

Attempted fused RMSNorm kernel to reduce E_2 kernel count. Result: **negative** —
kernel count increased from 1177 to 1282 (+105). Custom kernels break tinygrad's
graph fusion, causing surrounding ops to become separate kernels. Reverted.

## The fundamental bottleneck

The decode gap (34.2 vs 38.8 tok/s) and prefill gap (36 vs 2595 tok/s) are the
**same root cause**: 936 E_2 element-wise kernels per step, each with ~10 μs of
intra-graph overhead = ~9 ms of pure overhead.

These E_2 kernels are the MINIMUM set tinygrad's compiler can produce. They
represent:
- RMSNorm (cast + square → reduce → rsqrt + multiply + cast + weight)
- SwiGLU/FFN gating (silu + multiply)
- Residual additions
- Attention input processing (rope, normalize, reshape)

The compiler already fuses what it can (e.g., cast+square into one kernel,
rsqrt+multiply+cast+weight into one kernel). It cannot cross reduce boundaries
or fuse across custom kernel boundaries.

## Why custom kernels don't help

Tested: replacing nn.RMSNorm with a fused nv_rmsnorm custom kernel.

Result:
- nv_rmsnorm: 31 kernels replacing ~241 RMSNorm instances (good)
- BUT: nv_linear_q4_k went 37→54, nv_q8_quantize 51→71, total 1177→1282 (bad)
- Net: 33.73 tok/s vs 33.91 tok/s (slightly slower)

The custom kernel creates a hard boundary in the JIT graph. Ops that were
previously fused WITH the RMSNorm (e.g., the residual add after normalization)
become separate kernels. The fusion loss exceeds the kernel count savings.

## What would close the gap

1. **Compiler-level fusion** — teach tinygrad to fuse RMSNorm into the preceding
   GEMV kernel (like llama.cpp's fused QKV+RMSNorm). This requires engine work
   in tinygrad's codegen, not kernel work.

2. **Quantized GEMM for prefill** — the GEMV kernel processes any number of
   tokens in 0.073 ms/layer (weight-loading bound). But llama.cpp uses GEMM
   (tensor core matmul) for prefill, achieving 2595 tok/s. Writing a Q4_K GEMM
   kernel with tensor cores is a major project.

3. **Per-layer CUDA graphs** — currently all 64 layers are one graph. Per-layer
   graphs would allow intermediate memory reuse, enabling cs=8/16/32 without OOM.

## What DOESN'T help

- Custom RMSNorm/element-wise kernels (breaks fusion, net negative)
- More GEMV optimization (already 10× better per-token than llama.cpp)
- Flash attention at ctx=512 (no KV to tile over; helps at ctx>>4K)

## Current state (HEAD: `0f7bd750d`)

| Engine | Decode tok/s | Prefill tok/s (cs=2) | VRAM |
|---|---:|---:|---:|
| llama.cpp | 38.83 | 2595 | 15.65 GiB |
| tinygrad | 34.2 | 36.0 | 17.17 GiB |
| parity | 88% | 1.4% | comparable |

Decode is flat across ctx=512 to ctx=16384 (33.9 tok/s). The GEMV kernel is
excellent. The gap is entirely in element-wise overhead.
