# ACTIVE — Current state and next steps

> Snapshot: 2026-09-25. Evaluated on NVIDIA H100 SXM5 80GB (`g37` on Quartz). Decode reached 53.07 tok/s (18.84 ms/tok), a +38.0% speedup over 09-24 baseline (saving 7.15 ms/tok). 100% bit-exact top-5 logit match. Reduction grouping heuristic widened to 256 threads (`3a6346f48`). Fast 0.5B debugging model established.

## Read first

1. `handoff-2026-09-25-h100-baseline-and-quartz-setup.md` — **LATEST: Fast-start cheat sheet (one-command make targets), H100 SXM5 benchmark results, 2,060-kernel decode latency breakdown, 0.5B debugging setup, and reduction grouping optimization.**
2. `handoff-2026-09-24-l40s-cooperative-warp-and-kernel-census.md` — L40S benchmarks (31.10 tok/s), cooperative warp Q6_K implementation details.
3. `handoff-2026-09-24-decode-gemv-bandwidth-analysis-and-cooperative-warp.md` — GEMV memory load redundancy root cause analysis and cooperative warp architecture.
4. `handoff-2026-09-17-rmsnorm-experiment.md` — **CRITICAL NEGATIVE RESULT: why black-box custom RMSNorm kernels failed (broke compiler fusion, increased kernel count).**

## Fast-Start Commands (Root Makefile)

```bash
# On Quartz (ssh g37) or Lair (ssh lair-g6):
cd ~/projects/agent-handoffs   # (or /u/demistry/agent-handoffs on Lair)

make info            # Print auto-detected cluster, GPU, paths, and environment settings
make test-units      # Run both Q4_K and Q6_K cooperative warp unit sweeps (ALL OK)
make parity          # Verify 27B model logits (bit-exact top-5 match: [271, 25, 11751, 248044, 198])
make bench-tg        # Measure steady-state tinygrad decode throughput (512 ctx, 20 steps)
make bench-llama     # Measure reference llama.cpp decode throughput
make bench-tg-05b    # Fast-iteration test on 0.5B model (0.23s runtime, 86.95 tok/s)
make bench-llama-05b # Fast-iteration reference llama.cpp on 0.5B model (919 tok/s)
```

## Ground truth & Host Mappings

- **Active Compute Node**: `g37.quartz.uits.iu.edu` (`ssh g37`), NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s bandwidth).
- **Secondary Node**: `lair-g6` (`ssh lair-g6`), NVIDIA L40S 48GB GDDR6 (864 GB/s bandwidth).
- **tinygrad Workdir**:
  - Quartz: `$(HOME)/projects/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, HEAD `3a6346f48`).
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
| H100 SXM5 80GB (`g37`) | **tinygrad** (Coop Warp + r_256) | **53.07** | **18.84 ms** | **61.6%** | **Bit-exact match** |
| H100 SXM5 80GB (`g37`) | tinygrad (09-24 Baseline) | 38.47 | 25.99 ms | 44.6% | 0.9978 sim |
| **L40S 48GB** (`lair-g6`) | **llama.cpp** (`llama-bench`) | **37.95 ± 0.28** | **26.35 ms** | 100% | Reference |
| L40S 48GB (`lair-g6`) | **tinygrad** (Cooperative Warp) | **31.10** | **32.15 ms** | **82.0%** | 0.9980 sim |

## The Current Bottleneck: Where the 18.84 ms Goes on H100

Decode executes **2,060 kernels per step**:
1. **Quantized GEMV (497 kernels)**: **~10.4 ms** (streaming 16.8 GB weights at ~1.62 TB/s).
2. **Non-GEMV Overhead (1,563 kernels)**: **~8.4 ms**
   - Split 2-pass RMSNorm (`r_256_20` + `E_40_32_4`, 322 calls): **~4.8 ms**
   - Activation quantization (`nv_q8_quantize`, 257 calls): **~2.2 ms**
   - Residual additions & SwiGLU: **~1.4 ms**

llama.cpp launches only ~120 fused kernels per step with zero split reductions.

## Ranked Next Steps

1. **Full Single-Pass RMSNorm Reduction Fusion (~2.0–2.5 ms/tok savings on H100)**:
   - Eliminate the 322 split kernels (`r_256_20` and `E_40_32_4`) by fusing the reduction + elementwise scale into a single-pass block kernel without an intermediate global DRAM buffer.
   - Preserves surrounding elementwise fusion (do NOT insert a black-box custom UOp).
   - Expected result: Decode jumps from **53.1 tok/s $\rightarrow$ ~60+ tok/s**.
2. **Vectorize Cooperative Q4_K Loads (128-bit `uint4` / `v4.u32`) (~2.0 ms/tok savings)**:
   - Group loads into 128-bit vector transactions to boost memory bus saturation from 1.62 TB/s to 2.2+ TB/s.
3. **Activation Quantization Optimization**:
   - Streamline `nv_q8_quantize` or fuse quantization into preceding operations.
