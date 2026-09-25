# Handoff — H100 SXM5 Benchmark Baseline, Quartz Setup & Kernel Latency Breakdown (2026-09-25)

Prior handoffs:
- `handoff-2026-09-24-l40s-cooperative-warp-and-kernel-census.md`
- `handoff-2026-09-24-decode-gemv-bandwidth-analysis-and-cooperative-warp.md`
- `handoff-2026-09-17-rmsnorm-experiment.md` (CRITICAL: explains why black-box custom RMSNorm failed)
- `ACTIVE.md`

---

## 1. Quick-Start Cheat Sheet for Next Agent (NO EXPLORATION NEEDED)

Everything is pre-configured and automated in the root [`Makefile`](file:///Users/deven367/projects/agent-handoffs/Makefile).
Both clusters (Quartz and Lair) are fully supported with automatic cluster, GPU, and path detection.

### One-Command Operations
```bash
# Connect to current active compute node (H100 SXM5 80GB):
ssh g37

# Navigate to workspace:
cd ~/projects/agent-handoffs

# Check environment & detected paths:
make info

# Run all unit tests (Q4_K + Q6_K sweeps):
make test-units

# Verify numerical parity on full 27B model (bit-exact top-5 match):
make parity

# Measure steady-state tinygrad decode throughput (512 ctx, 20 steps):
make bench-tg

# Measure reference llama.cpp decode throughput:
make bench-llama
```

---

## 2. Cluster Ground Truth & Host Mappings

| Resource | Quartz (`ssh g37`) | Lair (`ssh lair-g6` / `ssh lair-g1`) |
|---|---|---|
| **GPU Hardware** | NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s) | L40S 48GB GDDR6 (`lair-g6`, 864 GB/s) / H100 NVL (`lair-g1`, 3,900 GB/s) |
| **Q4_K_M Model** | `/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` | `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` |
| **Q8_K_XL Model** | `/N/scratch/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf` | `/scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf` |
| **llama.cpp Dir** | `/N/slate/demistry/llama.cpp` | `$(HOME)/llama.cpp` |
| **llama-bench** | `/N/slate/demistry/llama.cpp/build/bin/llama-bench` | `/u/demistry/llama.cpp/build/bin/llama-bench` |
| **tinygrad-src** | `$(HOME)/projects/tinygrad-src` | `/u/demistry/tinygrad-src` |
| **CUDA Env** | `DEV=CUDA CUDA_PATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux` | `DEV=CUDA` (do NOT set `CUDA_PATH`) |
| **Git Branch** | `qwen27b-nv-q8-kernel` (HEAD `0fb19cf04`) | `qwen27b-nv-q8-kernel` |

---

## 3. Current Head-to-Head Benchmarks

### A. NVIDIA H100 SXM5 80GB (`g37` on Quartz)
*Evaluated on `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (15.65 GiB) at `ctx=512`, `f16` KV:*

| Engine | Decode tok/s | Decode ms/tok | Prefill pp512 tok/s | Top-5 Logit Match |
|---|---:|---:|---:|:---:|
| **llama.cpp** (`llama-bench`) | **86.18 ± 1.04** | **11.60 ms** | **2,440.75** | Reference |
| **tinygrad** (09-24 Baseline) | 38.47 | 25.99 ms | 36.0 | 0.9978 sim |
| **tinygrad** (Cooperative Warp) | **50.61** | **19.76 ms** | 36.0 | **Bit-exact match** |

- **Improvement**: **+31.5% speedup**, saving **6.23 ms/tok**!
- **Current Parity Ratio**: tinygrad is now at **58.7% of llama.cpp** (up from 44.6%).
- **Bit-Exact Logits**: Argmax 271; Top-5: `[(271, 34.3437), (25, 20.6735), (11751, 18.9752), (248044, 17.6756), (198, 16.6210)]`.

### B. NVIDIA L40S 48GB (`lair-g6` on Lair)
*Evaluated on the same model and context:*
- **llama.cpp**: **37.95 ± 0.28 tok/s (26.35 ms/tok)**
- **tinygrad**: **31.10 tok/s (32.15 ms/tok)**
- **Parity Ratio**: tinygrad is at **82.0% of llama.cpp** on L40S (5.80 ms/tok gap).

---

## 4. The Bottleneck: Where the 19.76 ms Goes on H100

From single-token hardware profiling (`DEBUG=2`) on H100 SXM5, decode executes **2,060 kernels per step**:

```text
Total decode time: 19.76 ms
├── 1. Quantized GEMV (497 kernels): ~10.4 ms
│   ├── nv_linear_q4_k (432 calls): ~8.8 ms (streaming weights at ~1.62 TB/s)
│   └── nv_linear_q6_k (65 calls): ~1.6 ms
└── 2. Non-GEMV Overhead (1,563 kernels): ~9.4 ms
    ├── Split RMSNorm (322 calls: r_16_320 + E_40_32_4): ~5.8 ms  <-- PRIMARY OPPORTUNITY
    ├── nv_q8_quantize (257 calls, memoized): ~2.2 ms
    ├── SwiGLU & Residuals (~880 calls: silu, mul, add): ~1.0 ms
    └── Flash attention & scan (64 calls): ~0.4 ms
```

**Why llama.cpp is faster (11.60 ms vs 19.76 ms):**
1. **GEMV streaming**: llama.cpp achieves ~2.2 TB/s effective bandwidth on H100 vs tinygrad's ~1.62 TB/s.
2. **Kernel count**: llama.cpp launches only ~120 fused kernels per step. Tinygrad launches **2,060 kernels**, spending ~9.4 ms purely on non-GEMV kernel invocations and DRAM round-trips!

---

## 5. Critical Lessons & Traps to Avoid

1. **DO NOT introduce a black-box custom RMSNorm UOp:**
   - Evaluated in `handoff-2026-09-17-rmsnorm-experiment.md`: replacing `nn.RMSNorm` with a standalone custom kernel created hard barriers in tinygrad's scheduler. Surrounding operations (residuals, activations) were forced into separate kernels, increasing total kernel count from 1,177 to 1,282 and slowing down execution.
   - Any RMSNorm optimization must be done via compiler/codegen fusion or lowering that preserves surrounding elementwise fusion.

2. **NVRTC CUDA Include Path on Quartz:**
   - On Quartz, CUDA headers are not in `/usr/local/cuda`. NVRTC requires `CUDA_PATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux` to find `<cuda_fp16.h>`.
   - On Lair, `CUDA_PATH` must NOT be set.
   - The root `Makefile` automatically handles this conditional setting via `$(TG_ENV)`.

3. **Bitwise Operations vs Integer Modulo in Custom Kernels:**
   - Never write `lane // 8` or `lane % 8` in custom UOp kernel loops. The tinygrad pattern matcher `pm_split_ranges` in `simplify.py` will split the 32-thread local range into `threadIdx.x=4, threadIdx.y=8`, causing a 2.5× memory slowdown due to destroyed coalescing.
   - Always use bitwise ops: `lane >> 3` and `lane & 7`.

---

## 6. Ranked Next Steps for Next Agent

### Priority 1: Fuse 2-Pass RMSNorm Reductions (~2.5–3.5 ms/tok gain on H100)
- **Problem**: In `tinygrad/nn/__init__.py`, `x.square().mean(-1)` on `dim=5120` lowers to two separate kernels: `r_16_320` (writes 16 intermediate floats to global DRAM) and `E_40_32_4` (reads from DRAM, computes rsqrt, scales by weight).
- **Solution**: Teach tinygrad's codegen or lowerer to recognize 1D reductions over hidden dimensions (e.g. 5,120) and fuse the reduction + scale into a single-pass block kernel using shared memory/warp shuffles, avoiding global DRAM allocation.
- **Expected Result**: Eliminates ~160 intermediate DRAM round-trips; saves **~3.0 ms/tok**, pushing H100 decode from **50.6 tok/s $\rightarrow$ ~60 tok/s**.

### Priority 2: Vectorize Cooperative Q4_K Memory Loads (`uint4` / 128-bit)
- **Problem**: Thread $t$ currently loads a single 32-bit word `raw[base + 4 + lane]`. While 32 threads coalesce into 128 bytes, NVIDIA memory pipelines achieve higher sustained bandwidth when individual threads issue 64-bit or 128-bit vector loads (`ld.global.v4.u32`).
- **Solution**: Have threads cooperatively load 128-bit chunks or pair adjacent blocks.
- **Expected Result**: Pushes GEMV memory bus saturation from 1.62 TB/s $\rightarrow$ 2.2+ TB/s; saves **~2.0 ms/tok**.

### Priority 3: Activation Quantization Pipeline
- **Problem**: `nv_q8_quantize` takes ~2.2 ms across 257 calls.
- **Solution**: Optimize the `nv_q8_quantize` kernel or explore fusing quantization directly into the preceding RMSNorm/elementwise kernel.
