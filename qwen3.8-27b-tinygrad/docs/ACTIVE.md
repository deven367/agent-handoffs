# ACTIVE — Current state and next steps

> Snapshot: 2026-09-25. Evaluated on NVIDIA H100 SXM5 80GB (`g37` on Quartz). Actions 1–3 landed (`c14c50207`) and **TODO 1–2 landed (`38342a3be`)**: Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize, **Fused Residual Add + RMSNorm (`nv_add_rmsnorm`, −64 E_* launches), Compact Q8 Activation Layout (36 B/group instead of 256 B)**. Parity verified bit-exact across all 248,320 vocabulary tokens. Unit tests 100% passing (`make test-units`).

## Read first

1. `handoff-2026-09-25-fused-add-rmsnorm-and-compact-q8.md` — **LATEST: TODO 1+2 Landed (`38342a3be`): fused add+RMSNorm, compact Q8 layout, 73.69 tok/s, clean A/B re-measure, benchmark-hygiene finding, GEMV output-buffer lever.**
2. `handoff-2026-09-25-actions-1-2-3-completed.md` — Actions 1, 2, and 3 Landed (Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize), Current Gap Analysis, and Unrolling Trap Takeaways (`c14c50207`).
3. `handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md` — GatedDeltaNet Normalization Profiling, GPU Clock Dynamics, and Next Optimization Roadmap.
4. `handoff-2026-09-25-task3-fused-rmsnorm.md` — Task 3 Single-Pass Block-Fused RMSNorm (eliminates 256 reduction kernels, saving 1.57 ms/tok).
5. `handoff-2026-09-25-task2-vectorized-q4k-gemv.md` — Task 2 Vectorized Cooperative GEMV (62.06 tok/s on H100, 128-bit/64-bit vector loads).
6. `handoff-2026-09-25-h100-baseline-and-quartz-setup.md` — Fast-start cheat sheet (one-command make targets), H100 SXM5 benchmark results, 2,060-kernel decode latency breakdown, 0.5B debugging setup.

## Fast-Start Commands (Root Makefile)

```bash
# On Quartz (ssh g37) or Lair (ssh lair-g6):
cd ~/projects/agent-handoffs   # (or /u/demistry/agent-handoffs on Lair)

make info            # Print auto-detected cluster, GPU, paths, and environment settings
make test-units      # Run both Q4_K and Q6_K cooperative warp unit sweeps (ALL OK)
make parity          # Verify 27B model logits (top-5 match: [271, 25, 11751, 248044, 198])
make bench-tg        # Measure steady-state tinygrad decode throughput (512 ctx, 20 steps)
make bench-llama     # Measure reference llama.cpp decode throughput (85.87 tok/s clean)
make bench-tg-05b    # Fast-iteration test on 0.5B model (0.18s runtime, 108.33 tok/s)
make bench-llama-05b # Fast-iteration reference llama.cpp on 0.5B model (919 tok/s)
```

## Ground truth & Host Mappings

- **Active Compute Node**: `g37.quartz.uits.iu.edu` (`ssh g37`), NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s bandwidth). Slurm-gated (needs an active job, partition `h100-debu`, 1 h limit — `scontrol requeue` to renew).
- **Secondary Node**: `lair-g6` (`ssh lair-g6`), NVIDIA L40S 48GB GDDR6 (864 GB/s bandwidth).
- **tinygrad Workdir**:
  - Quartz: `$(HOME)/projects/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, HEAD `38342a3be`).
  - Lair: `/u/demistry/tinygrad-src` (branch `qwen27b-nv-q8-kernel`).
- **Model Paths**:
  - 27B Q4_K_M (Quartz): `/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
  - 0.5B Q4_K_M (Quartz): `/N/scratch/demistry/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
  - 27B Q4_K_M (Lair): `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
- **Launcher**: `/N/u/demistry/Quartz/projects/agent-handoffs/Makefile` (host-aware, auto-selects cluster settings).

## Benchmark Comparison Table (Q4_K_M, ctx=512, f16 KV)

| GPU / Platform | Engine | Decode tok/s | Decode ms/tok | Parity Ratio | Logit Parity |
|---|---|---:|---:|:---:|:---:|
| **H100 SXM5 80GB** (`g37`) | **llama.cpp** (`llama-bench`, clean) | **85.87 ± 1.00** | **11.65 ms** | 100% | Reference |
| **H100 SXM5 80GB** (`g37`) | **tinygrad (TODO 1+2: fused add_rmsnorm, compact q8, `38342a3be`)** | **73.69** | **13.57 ms** | **85.8%** | **Bit-exact match** |
| H100 SXM5 80GB (`g37`) | tinygrad (Actions 1–3, `c14c50207`) | 68.79 | 14.54 ms | 79.8% | Match (diff $\le 0.0019$) |
| H100 SXM5 80GB (`g38`) | tinygrad (Task 2: Vectorized Coop) | 62.06 | 16.11 ms | 72.0% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (Task 1: Heuristic r_256) | 53.07 | 18.84 ms | 61.6% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (09-24 Baseline) | 38.47 | 25.99 ms | 44.6% | 0.9978 sim |
| **L40S 48GB** (`lair-g6`) | **llama.cpp** (`llama-bench`) | **37.95 ± 0.28** | **26.35 ms** | 100% | Reference |
| L40S 48GB (`lair-g6`) | **tinygrad** (Cooperative Warp) | **31.10** | **32.15 ms** | **82.0%** | 0.9980 sim |

## Latency Breakdown & Current Status on H100 (`38342a3be`)

```text
Total decode time: ~13.6 ms/tok (73.7 tok/s steady)   [GEMV split from `c14c50207` profile;
                                                       non-GEMV estimates — re-run kstat.py]
├── 1. Quantized GEMV (572 kernels): ~7.2 ms
│   ├── Q4_K v4 (444 calls, 128-bit vectorized): ~5.2 ms
│   ├── Q4_K v2 (64 calls, 64-bit vectorized): ~0.9 ms
│   └── Q6_K v2 (64 calls, 64-bit vectorized, 1.23x speedup): ~1.1 ms
└── 2. Non-GEMV Overhead (~1,424 kernels, ~6.4 ms)
    ├── Single-Pass RMSNorm (nv_rmsnorm, 129 calls): ~1.4 ms  (64 calls fused into add_rmsnorm)
    ├── Fused Add+RMSNorm (nv_add_rmsnorm, 64 calls, NEW): ~0.7 ms
    ├── Fused QK L2 Norm (nv_normalize, 128 calls): ~0.7 ms
    ├── Activation Quantization (nv_q8_quantize, 192 calls, 36 B/group writes): ~1.2 ms
    └── SwiGLU, RoPE & Residual Additions (~276 calls, 64 E_* adds gone): ~2.2 ms
```

## Ranked Next Steps to Close the Remaining ~1.9 ms Gap to llama.cpp (11.65 ms)

1. **Lever A: Compact GEMV output buffers** (largest structural waste remaining):
   - Every GEMV stores `(rows, 32)` f32 per output row (128 B written, only word 0 read). ~30–40 MB/tok of write waste across 572 calls; up to ~1 ms/tok. Touches all three GEMV families.
2. **TODO 3: Multi-Warp Grid-Fused RMSNorm + Q8 Quantize** (~0.5–0.8 ms/tok):
   - Use a multi-warp grid launch (160 warps) instead of a single-warp loop to avoid Python UOp unrolling register spilling.
3. **Fused 2nd residual add (`h + ffn_out`) into the next block's `attn_norm`**:
   - Cross-block restructure of `Transformer.forward` (yield `(residual, normed)` pairs); removes ~64 more E_* launches.
4. **Re-profile first** (kstat.py on a DEBUG=2 log) to re-rank after `38342a3be`.

**Benchmark hygiene:** a resident `llama-server` (44% SM bursts) cut llama-bench 85.87 → 48.51 tok/s. Always `make stop` before benchmarking, `make serve` after.
