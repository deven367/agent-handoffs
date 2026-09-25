# ACTIVE — Current state and next steps

> Snapshot: 2026-09-25. Evaluated on NVIDIA H100 SXM5 80GB (`g38` on Quartz). Decode reached **68.79 tok/s (14.54 ms/tok)**, an overall **+78.8% speedup** over 09-24 baseline (saving 11.45 ms/tok). Top-5 logit parity verified with diff $\le 0.0019$ across all 248,320 vocabulary tokens. Task 3 Single-Pass Block-Fused RMSNorm landed (`2132231dd`). Task 2 vectorized cooperative GEMV landed (`c162d326b`). Task 1 reduction heuristic widened (`3a6346f48`).

## Read first

1. `handoff-2026-09-25-task3-fused-rmsnorm.md` — **LATEST: Task 3 Single-Pass Block-Fused RMSNorm (68.79 tok/s on H100, eliminates 256 reduction kernels, saving 1.57 ms/tok).**
2. `handoff-2026-09-25-task2-vectorized-q4k-gemv.md` — Task 2 Vectorized Cooperative GEMV (62.06 tok/s on H100, 128-bit/64-bit vector loads).
3. `handoff-2026-09-25-h100-baseline-and-quartz-setup.md` — Fast-start cheat sheet (one-command make targets), H100 SXM5 benchmark results, 2,060-kernel decode latency breakdown, 0.5B debugging setup, and reduction grouping optimization (Task 1).
4. `handoff-2026-09-24-l40s-cooperative-warp-and-kernel-census.md` — L40S benchmarks (31.10 tok/s), cooperative warp Q6_K implementation details.
5. `handoff-2026-09-17-rmsnorm-experiment.md` — Why earlier black-box custom RMSNorm attempts failed (lack of activation caching and un-memoized custom kernel boundaries).

## Fast-Start Commands (Root Makefile)

```bash
# On Quartz (ssh g38) or Lair (ssh lair-g6):
cd ~/projects/agent-handoffs   # (or /u/demistry/agent-handoffs on Lair)

make info            # Print auto-detected cluster, GPU, paths, and environment settings
make test-units      # Run both Q4_K and Q6_K cooperative warp unit sweeps (ALL OK)
make parity          # Verify 27B model logits (top-5 match: [271, 25, 11751, 248044, 57590])
make bench-tg        # Measure steady-state tinygrad decode throughput (512 ctx, 20 steps -> 68.79 tok/s)
make bench-llama     # Measure reference llama.cpp decode throughput (86.18 tok/s)
make bench-tg-05b    # Fast-iteration test on 0.5B model (0.18s runtime, 108.33 tok/s)
make bench-llama-05b # Fast-iteration reference llama.cpp on 0.5B model (919 tok/s)
```

## Ground truth & Host Mappings

- **Active Compute Node**: `g38.quartz.uits.iu.edu` (`ssh g38`), NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s bandwidth).
- **Secondary Node**: `lair-g6` (`ssh lair-g6`), NVIDIA L40S 48GB GDDR6 (864 GB/s bandwidth).
- **tinygrad Workdir**:
  - Quartz: `$(HOME)/projects/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, HEAD `2132231dd`).
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

## Latency Breakdown of Current 14.54 ms/tok on H100

```text
Total decode time: 14.54 ms
├── 1. Quantized GEMV (572 kernels): ~7.7 ms
│   ├── Q4_K v4 (444 calls, 128-bit vectorized): ~5.2 ms
│   ├── Q4_K v2 (64 calls, 64-bit vectorized): ~0.9 ms
│   └── Q6_K (64 calls, 32-bit cooperative): ~1.6 ms
└── 2. Non-GEMV Overhead (1,044 kernels): ~6.8 ms
    ├── Single-Pass RMSNorm (nv_rmsnorm, 257 calls): ~2.6 ms  (down from 4.8 ms)
    ├── Activation Quantization (nv_q8_quantize, 192 calls): ~1.8 ms
    └── SwiGLU, RoPE & Residual Additions (~595 calls): ~2.4 ms
```

## Ranked Next Steps to Close the 2.94 ms Gap to llama.cpp

1. **Q6_K Vectorization (`tinygrad/llm/kernels/nv_q6k.py`)**:
   - 64 calls taking ~1.6 ms. Vectorize the cooperative warp with 64-bit loads (`uint2`) following the Task 2 pattern. Expected savings: **~0.5 ms/tok**.
2. **Fused Residual Add + RMSNorm (`fused_add_rmsnorm`)**:
   - Fuse `h = x + attn_output` directly into the input of `ffn_norm` to eliminate intermediate elementwise kernels. Expected savings: **~0.6 ms/tok**.
3. **Activation Quantization Optimization (`nv_q8_quantize`)**:
   - 192 calls taking ~1.8 ms. Vectorize stores and reduce dispatch overhead. Expected savings: **~0.5 ms/tok**.
