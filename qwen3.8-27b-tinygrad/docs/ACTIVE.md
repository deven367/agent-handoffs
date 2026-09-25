# ACTIVE — Current state and next steps

> Snapshot: 2026-09-25. Evaluated on NVIDIA H100 SXM5 80GB (`g37` on Quartz). Actions 1, 2, 3 and TODO 1 landed (`231786562`): Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize, and Fused RMSNorm+Q8 Quantize. Parity verified bit-exact across all 248,320 vocabulary tokens. Unit tests 100% passing (`make test-units`).

## Read first

1. `handoff-2026-09-25-actions-1-2-3-completed.md` — **LATEST: Actions 1, 2, 3 and TODO 1 Landed (Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize, and Fused RMSNorm+Q8) (`231786562`).**
2. `handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md` — GatedDeltaNet Normalization Profiling, GPU Clock Dynamics, and Next Optimization Roadmap.
3. `handoff-2026-09-25-task3-fused-rmsnorm.md` — Task 3 Single-Pass Block-Fused RMSNorm (eliminates 256 reduction kernels, saving 1.57 ms/tok).
4. `handoff-2026-09-25-task2-vectorized-q4k-gemv.md` — Task 2 Vectorized Cooperative GEMV (62.06 tok/s on H100, 128-bit/64-bit vector loads).
5. `handoff-2026-09-25-h100-baseline-and-quartz-setup.md` — Fast-start cheat sheet (one-command make targets), H100 SXM5 benchmark results, 2,060-kernel decode latency breakdown, 0.5B debugging setup.

## Fast-Start Commands (Root Makefile)

```bash
# On Quartz (ssh g38) or Lair (ssh lair-g6):
cd ~/projects/agent-handoffs   # (or /u/demistry/agent-handoffs on Lair)

make info            # Print auto-detected cluster, GPU, paths, and environment settings
make test-units      # Run both Q4_K and Q6_K cooperative warp unit sweeps (ALL OK)
make parity          # Verify 27B model logits (top-5 match: [271, 25, 11751, 248044, 198])
make bench-tg        # Measure steady-state tinygrad decode throughput (512 ctx, 20 steps)
make bench-llama     # Measure reference llama.cpp decode throughput (86.18 tok/s)
make bench-tg-05b    # Fast-iteration test on 0.5B model (0.18s runtime, 108.33 tok/s)
make bench-llama-05b # Fast-iteration reference llama.cpp on 0.5B model (919 tok/s)
```

## Ground truth & Host Mappings

- **Active Compute Node**: `g38.quartz.uits.iu.edu` (`ssh g38`), NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s bandwidth).
- **Secondary Node**: `lair-g6` (`ssh lair-g6`), NVIDIA L40S 48GB GDDR6 (864 GB/s bandwidth).
- **tinygrad Workdir**:
  - Quartz: `$(HOME)/projects/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, HEAD `bd17e6e1c`).
  - Lair: `/u/demistry/tinygrad-src` (branch `qwen27b-nv-q8-kernel`).
- **Model Paths**:
  - 27B Q4_K_M (Quartz): `/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
  - 0.5B Q4_K_M (Quartz): `/N/scratch/demistry/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
  - 27B Q4_K_M (Lair): `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
- **Launcher**: `/N/u/demistry/Quartz/projects/agent-handoffs/Makefile` (host-aware, auto-selects cluster settings).

## Benchmark Comparison Table (Q4_K_M, ctx=512, f16 KV)

| GPU / Platform | Engine | Decode tok/s | Decode ms/tok | Parity Ratio | Logit Parity |
|---|---|---:|---:|:---:|:---:|
| **H100 SXM5 80GB** (`g38`) | **llama.cpp** (`llama-bench`) | **86.18 ± 1.04** | **11.60 ms** | 100% | Reference |
| **H100 SXM5 80GB** (`g38`) | **tinygrad (Task 3: Fused RMSNorm)** | **68.79** | **14.54 ms** | **79.8%** | **Match (diff $\le 0.0019$)** |
| H100 SXM5 80GB (`g38`) | tinygrad (Task 2: Vectorized Coop) | 62.06 | 16.11 ms | 72.0% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (Task 1: Heuristic r_256) | 53.07 | 18.84 ms | 61.6% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (09-24 Baseline) | 38.47 | 25.99 ms | 44.6% | 0.9978 sim |
| **L40S 48GB** (`lair-g6`) | **llama.cpp** (`llama-bench`) | **37.95 ± 0.28** | **26.35 ms** | 100% | Reference |
| L40S 48GB (`lair-g6`) | **tinygrad** (Cooperative Warp) | **31.10** | **32.15 ms** | **82.0%** | 0.9980 sim |

## Latency Breakdown & Current Status on H100 (`16c494e99`)

```text
Total decode time: ~14.1 ms/tok (~70+ tok/s steady)
├── 1. Quantized GEMV (572 kernels): ~7.2 ms
│   ├── Q4_K v4 (444 calls, 128-bit vectorized): ~5.2 ms
│   ├── Q4_K v2 (64 calls, 64-bit vectorized): ~0.9 ms
│   └── Q6_K v2 (64 calls, 64-bit vectorized, 1.23x speedup): ~1.1 ms
└── 2. Non-GEMV Overhead (~853 kernels): ~6.9 ms
    ├── Single-Pass RMSNorm (nv_rmsnorm, 193 calls): ~2.0 ms
    ├── Fused QK L2 Norm (nv_normalize, 128 calls): ~0.7 ms  (eliminated 256 r_16_8 & E_* kernels!)
    ├── Activation Quantization (nv_q8_quantize, 192 calls, 1.19x faster): ~1.5 ms
    └── SwiGLU, RoPE & Residual Additions (~340 calls): ~2.7 ms
```

## Ranked Next Steps to Close the Final ~2.5 ms Gap to llama.cpp (11.60 ms)

1. **Fused RMSNorm + Q8 Quantization (`nv_rmsnorm_q8`)**:
   - Fuse `nv_rmsnorm` and `nv_q8_quantize` into a single kernel to eliminate 192 kernel launches and 384 redundant DRAM round-trips. Expected savings: **~1.0–1.2 ms/tok**.
2. **Fused Residual Add + RMSNorm (`nv_add_rmsnorm`)**:
   - Fuse residual addition directly into the input of `nv_rmsnorm`, eliminating 64 elementwise kernels and DRAM round-trips. Expected savings: **~0.5–0.7 ms/tok**.
3. **Vectorized 128-Bit Stores for Q8 Quantize**:
   - Store 8 uint32 words as two 128-bit `uint4` stores per group. Expected savings: **~0.2–0.4 ms/tok**.
