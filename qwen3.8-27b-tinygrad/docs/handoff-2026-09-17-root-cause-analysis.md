# Handoff 2026-09-17: Why tinygrad trails llama.cpp — root causes & evidence

Consolidates the H100 (quartz) session. Prior L40S findings: `handoff-2026-09-17-parity-benchmark.md`.

## Headline numbers

Single-run, same model (`Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`, 16.81 GB), `max_context=512`.

| GPU | engine | decode (tg) | prefill (pp) |
|---|---|---:|---:|
| L40S 46 GB | llama.cpp | 38.83 tok/s | 2595 tok/s |
| L40S 46 GB | tinygrad | 34.2 tok/s | 36.0 tok/s (cs=2) |
| H100 80 GB | llama.cpp | 80.88 tok/s | 2424 tok/s |
| H100 80 GB | tinygrad | **42.11 tok/s** | **38.7 tok/s** (cs=2) |

Decode gap: **1.13× on L40S**, **1.92× on H100**. Prefill gap: **72× on L40S**, **63× on H100**.

**Prefill is warmup-sensitive.** One 256-token prefill (cs=2), six consecutive runs, H100:
`5.5 → 38.7 → 38.7 → 38.7 → 38.7 → 38.7` tok/s (6609–6611 ms each after the first).
Run 0 (5.5 tok/s, 46.4 s) is JIT compilation. Timing only the *second* run gives 16.4 tok/s —
an artifact of the clock/allocator ramp. Take ≥2 warmup prefills before timing.

## Root cause A — prefill: tinygrad has no working tensor-core matmul on CUDA

Measured directly (`bench_matmul_tc.py`, 512×5120 @ 5120×5120 fp16, H100):

| config | TFLOPS |
|---|---:|
| `TC=1 TC_OPT=0` (default) | **6.4** |
| `TC=1 TC_OPT=0 BEAM=2` | 3.9 |
| `TC=2` (TC shape, no WMMA) | 1.7 |

H100 fp16 tensor-core peak ≈ 990 TFLOPS. tinygrad achieves **0.6% of peak**.

Evidence that tensor cores are essentially never emitted: of 414 kernels in the CUDA compile
cache, **exactly 1** contains `mma.sync`.

Mechanism: `UOp.wmma` is only constructed in `tinygrad/llm/kernels/amd.py` (AMD LLM kernels).
The compiler *can* render it (`renderer/cstyle.py::CUDARenderer.render_kernel` emits
`mma.sync.aligned`), and `codegen/opt/tc.py::get_cuda` supplies sm75/80/89 shapes, and
`codegen/opt/postrange.py::_apply_tc_opt` applies them — but for the model's linears the
custom GEMV kernels are used instead, so no matmul is ever scheduled.

llama.cpp prefill reaches 2424 tok/s because cuBLAS batched GEMM runs on tensor cores.

## Root cause B — decode: fixed per-kernel overhead, not bandwidth

Bandwidth alone predicts **220 tok/s** on H100 (3350 GB/s ÷ 15.2 GB). tinygrad gets 42.1.
Going L40S → H100 multiplies bandwidth by **3.9×** but decode by only **1.23×**.

=> decode is bound by the ~1400-kernel-per-token graph, not by memory. Each of the 936
`E_2` element-wise kernels costs ~10 μs of inter-kernel latency inside the CUDA graph.

llama.cpp at 80.9 tok/s (H100) = 12.4 ms/token; tinygrad 23.8 ms/token.

## What was tried, and the measured outcome

1. **Q4_K dequant → fp16 → tensor-core-less matmul for prefill** (custom `nv_q4k_dequant.py`
   kernel producing a (out,in) fp16 matrix, then `x.dot(w.T)`).
   - Correctness: **verified** — identical greedy tokens `[271, 550, 220, 17, 13, 17, 13, 17]`
     vs the GEMV path.
   - Speed: **no gain** — 38.6 vs 38.7 tok/s (cs=2). The matmul is not the bottleneck at cs=2.
   - VRAM: needs packed (15.65 GB) + fp16 (~54 GB) ≈ 66–70 GB. Only fits on H100.
   - Reverted: provides no benefit, costs 50 GB.

2. **Keeping fp16 weights via model-load-time packing** (pack first, then realize fp16).
   - Worked (66 GB resident, correct output) but same speed. Reverted with (1).

3. **Larger chunk sizes** (fresh process per cs, one warmup, H100, GEMV):

   | cs | tok/s |
   |---:|---:|
   | 2 | 38.7 (steady state, see above) |
   | 8 | 16.6 |
   | 16 | 16.3 |
   | 32 | OOM (78.02 GB) |

   cs=2 is the optimum; cs=32 exhausts 80 GB. (The cs=8/16 numbers come from single-warmup runs
   and are likely understated by the same ramp artifact; the OOM at cs=32 is not affected.)

4. **`JIT_BATCH_SIZE=0`** (single CUDA graph instead of 2 batches): 33.45 vs 33.86 tok/s — slower.

5. **`JIT=1`** (no graph capture): 13.1 tok/s prefill — no better.

6. **`BEAM=2`**: 34.71 vs 33.86 tok/s decode (+2.5%) but 3× JIT compile time (184 s vs 64 s).

7. **Fused RMSNorm custom kernel** (prior session): kernel count *increased* 1177→1282 because
   custom kernels are fusion barriers. Reverted.

## Bug found and fixed this session

`flash_attention` requires `T_pad % 32 == 0`. The NV gate added it for *all* token counts, so any
chunked prefill (cs=2/4/8/16) crashed with `AssertionError: chunk_size must be a multiple of 32`.
Fix: gate on `resolve(T == 1)` — commit `3c92cf1a4` on `fork/qwen27b-nv-q8-kernel`.
Verified: decode and prefill both run, block-level bisect matches (rel=0.00e+00).

## What would actually close the gaps

**Prefill (63–72×):**
- A CUDA Q4_K GEMM that dequantizes into registers and feeds `mma.sync` — i.e. port
  `amd.py::_q5_linear_f16_wmma_kernel` to NV. It uses `UOp.wmma` (renderer-supported) but its
  layout helpers (`_wmma_layout`, `_wmma_stores`) call AMD-only builtins
  (`__builtin_amdgcn_mbcnt_lo`, `__builtin_amdgcn_ds_swizzle`) and assume AMD fragment layout.
  A CUDA port needs the m16n8k16 fragment mapping rewritten.
- AND raising the chunk size, which needs the SSM/attention intermediates to stop being retained
  for the whole JIT graph (cs=32 wants >78 GB for a 17 GB model).

**Decode (1.1–1.9×):**
- Fewer kernels. 936 `E_2` ops/token must fuse. Custom kernels make this worse (measured twice).

Neither is a kernel-level tweak; both are compiler/engine work.

## Environment (quartz / H100)

```bash
ssh quartz 'salloc --account=r00117 --partition=h100-single --gres=gpu:1 --mem=128G -t 4:00:00 bash'
# then: ssh -o ProxyJump=quartz demistry@<node>   (config alias: qgpu)
```
- tinygrad at `/N/slate/demistry/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, HEAD `3c92cf1a4`), `pip install -e`.
- Model at `/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
  (`hf download OBLITERATUS/Qwen3.8-27B-OBLITERATED --include "Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"`).
- **Required env**: `CUDA_PATH=/N/soft/rhel8/cuda/12.6 DEV=CUDA` (NVRTC cannot find `cuda_fp16.h` otherwise).
- llama.cpp: `/N/slate/demistry/llama.cpp/build/bin/llama-bench`, needs
  `LD_LIBRARY_PATH=build/bin:/N/soft/rhel8/cuda/12.6/targets/x86_64-linux/lib:/N/soft/rhel8/gcc/14.2.0/lib64`.
- **Allocation must request ≥128 GB host RAM**: default `mem=4000M` is killed during model load.
- Benchmarks need a *full* warmup at the target chunk size; a short warmup under-reports prefill
  by ~2.5× (13 vs 38.7 tok/s).
