# ACTIVE — Current state and next steps

> Snapshot: 2026-09-25. Evaluated on NVIDIA H100 SXM5 80GB (`g37` on Quartz). Decode reached 62.06 tok/s (16.11 ms/tok), a +61.3% speedup over 09-24 baseline (saving 9.88 ms/tok). 100% bit-exact top-5 logit match. Task 2 vectorized cooperative GEMV (uint4/uint2) landed (`c162d326b`). Task 1 reduction heuristic widened (`3a6346f48`). Fast 0.5B debugging model established.

## Read first

1. `handoff-2026-09-25-task2-vectorized-q4k-gemv.md` — **LATEST: Task 2 Vectorized Cooperative GEMV (62.06 tok/s on H100, 128-bit/64-bit vector loads, +17% speedup over Task 1).**
2. `handoff-2026-09-25-h100-baseline-and-quartz-setup.md` — Fast-start cheat sheet (one-command make targets), H100 SXM5 benchmark results, 2,060-kernel decode latency breakdown, 0.5B debugging setup, and reduction grouping optimization (Task 1).
3. `handoff-2026-09-24-l40s-cooperative-warp-and-kernel-census.md` — L40S benchmarks (31.10 tok/s), cooperative warp Q6_K implementation details.
4. `handoff-2026-09-24-decode-gemv-bandwidth-analysis-and-cooperative-warp.md` — GEMV memory load redundancy root cause analysis and cooperative warp architecture.
5. `handoff-2026-09-17-rmsnorm-experiment.md` — **CRITICAL NEGATIVE RESULT: why black-box custom RMSNorm kernels failed (broke compiler fusion, increased kernel count).**

## Fast-Start Commands (Root Makefile)

```bash
# On Quartz (ssh g37) or Lair (ssh lair-g6):
cd ~/projects/agent-handoffs   # (or /u/demistry/agent-handoffs on Lair)

make info            # Print auto-detected cluster, GPU, paths, and environment settings
make test-units      # Run both Q4_K and Q6_K cooperative warp unit sweeps (ALL OK)
make parity          # Verify 27B model logits (bit-exact top-5 match: [271, 25, 11751, 248044, 198])
make bench-tg        # Measure steady-state tinygrad decode throughput (512 ctx, 20 steps)
make bench-llama     # Measure reference llama.cpp decode throughput
make bench-tg-05b    # Fast-iteration test on 0.5B model (0.23s runtime, 86.10 tok/s)
make bench-llama-05b # Fast-iteration reference llama.cpp on 0.5B model (919 tok/s)
```

## Ground truth & Host Mappings

- **Active Compute Node**: `g37.quartz.uits.iu.edu` (`ssh g37`), NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s bandwidth).
- **Secondary Node**: `lair-g6` (`ssh lair-g6`), NVIDIA L40S 48GB GDDR6 (864 GB/s bandwidth).
- **tinygrad Workdir**:
  - Quartz: `$(HOME)/projects/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, HEAD `c162d326b`).
  - Lair: `/u/demistry/tinygrad-src` (branch `qwen27b-nv-q8-kernel`).
- **Model Paths**:
  - 27B Q4_K_M (Quartz): `/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
  - 0.5B Q4_K_M (Quartz): `/N/scratch/demistry/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
  - 27B Q4_K_M (Lair): `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
- **Launcher**: `/N/u/demistry/Quartz/projects/agent-handoffs/Makefile` (host-aware, auto-selects cluster settings).

## Benchmark Comparison Table (Q4_K_M, ctx=512, f16 KV)

| GPU / Platform | Engine | Decode tok/s | Decode ms/tok | Parity Ratio | Logit Parity |
|---|---|---:|---:|:---:|:---:|
| **H100 SXM5 80GB** (`g37`) | **llama.cpp** (`llama-bench`) | **86.18 ± 1.04** | **11.60 ms** | 100% | Reference |
| **H100 SXM5 80GB** (`g37`) | **tinygrad (Task 2: Vectorized Coop)** | **62.06** | **16.11 ms** | **72.0%** | **Bit-exact match** |
| H100 SXM5 80GB (`g37`) | tinygrad (Task 1: Heuristic r_256) | 53.07 | 18.84 ms | 61.6% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (09-24 Baseline) | 38.47 | 25.99 ms | 44.6% | 0.9978 sim |
| **L40S 48GB** (`lair-g6`) | **llama.cpp** (`llama-bench`) | **37.95 ± 0.28** | **26.35 ms** | 100% | Reference |
| L40S 48GB (`lair-g6`) | **tinygrad** (Cooperative Warp) | **31.10** | **32.15 ms** | **82.0%** | 0.9980 sim |

## The Current Bottleneck: Where the 16.11 ms Goes on H100

With Task 2 (128-bit/64-bit vectorization) landed, GEMV latency dropped by ~2.74 ms/tok:
1. **Quantized GEMV (497 kernels)**: **~7.7 ms** (streaming 16.8 GB weights at ~2.18 TB/s effective bandwidth).
2. **Non-GEMV Overhead (1,563 kernels)**: **~8.4 ms**
   - Split 2-pass RMSNorm (`r_256_20` + `E_40_32_4`, 322 calls): **~4.8 ms**  <-- NOW THE DOMINANT BOTTLENECK
   - Activation quantization (`nv_q8_quantize`, 257 calls): **~2.2 ms**
   - Residual additions & SwiGLU: **~1.4 ms**

llama.cpp launches only ~120 fused kernels per step with zero split reductions and zero DRAM round-trips for norms.

## Ranked Next Steps

1. **Full Single-Pass RMSNorm Reduction Fusion (~2.0–2.5 ms/tok savings on H100)**:
   - Eliminate the 322 split kernels (`r_256_20` and `E_40_32_4`) by fusing the reduction + elementwise scale into a single-pass block kernel without an intermediate global DRAM buffer.
   - Preserves surrounding elementwise fusion (do NOT insert a black-box custom UOp, see `handoff-2026-09-17-rmsnorm-experiment.md`).
   - Expected result: Decode jumps from **62.06 tok/s $\rightarrow$ ~70–75 tok/s** (closing within 1.5 ms of llama.cpp).
2. **Activation Quantization Optimization (`nv_q8_quantize`)**:
   - 257 calls taking ~2.2 ms. Optimize threadblock shape or fuse quantization into preceding activation layers.
3. **Q6_K Vectorization**:
   - Apply vector loads to `nv_q6k.py` to optimize the remaining 65 Q6_K layers.
