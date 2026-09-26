# ACTIVE — Current state and next steps

> Snapshot: 2026-09-25 (End-of-day / Bedtime). Evaluated on NVIDIA H100 SXM5 80GB (`g37` on Quartz). **Vectorized 128-bit Q4_K Scales & GEMV Landed (`4026d67ff`)**: In addition to prior landings (Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize, Fused Residual Add + RMSNorm, Compact Q8 Activation Layout, Two-Stage Custom Greedy Argmax `nv_argmax`, and Fused SwiGLU SiLU*up), all Q4_K decode kernels (`_q4_k_v4_decode_kernel`, `_q4_k_v2_decode_kernel`, `_q4_k_decode_kernel`) now load the 16-byte block header (`d`, `dmin`, scales 0..11) via a single 128-bit `uint4` memory read (`ld.global.v4.u32`) and unpack scales entirely in registers using ALU shifts and bitmasks, completely eliminating up to 136 scalar memory loads per warp. Steady-state decode throughput broke past 85 tok/s, reaching **85.31 tok/s (11.72 ms/tok)**, closing **99.0%** of the gap to `llama.cpp` reference (**86.18 ± 1.04 tok/s, 11.60 ms/tok**). The remaining gap is now just **0.12 ms/tok**, officially placing tinygrad inside llama.cpp's 1-sigma uncertainty window ($[85.14, 87.22]$ tok/s). Parity verified bit-exact across all 248,320 vocabulary tokens (`make parity`), and greedy token rollout verified identical (`make token-ab`). Unit tests 100% passing (`make test-units`).

## Read first

1. `handoff-2026-09-25-vectorized-q4k-scales-landed.md` — **START HERE: 128-bit vectorized Q4_K scale loading, ALU in-register unpacking, benchmark results (85.31 tok/s), and final 0.12 ms roadmap.**
2. `handoff-2026-09-25-nv-argmax-landed.md` — custom two-stage nv_argmax kernel implementation, SwiGLU SiLU*up fusion, and 83.76 tok/s baseline.
3. `handoff-2026-09-25-next-agent-argmax-kernel.md` — prior handoff proposing the argmax kernel and analyzing the 1.5 ms bottleneck.
4. `handoff-2026-09-25-lever-a-rejected-and-reprofile.md` — evidence: Lever A rejected (PTX root cause, zero copy kernels), fresh kernel census, and the two failed argmax restagings.
5. `handoff-2026-09-25-fused-add-rmsnorm-and-compact-q8.md` — TODO 1+2 Landed (`38342a3be`): fused add+RMSNorm, compact Q8 layout, 73.69 tok/s, clean A/B re-measure, benchmark-hygiene finding.
6. `handoff-2026-09-25-actions-1-2-3-completed.md` — Actions 1, 2, and 3 Landed (Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize), Current Gap Analysis, and Unrolling Trap Takeaways (`c14c50207`).
7. `handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md` — GatedDeltaNet Normalization Profiling, GPU Clock Dynamics, and Next Optimization Roadmap.
8. `handoff-2026-09-25-task3-fused-rmsnorm.md` — Task 3 Single-Pass Block-Fused RMSNorm (eliminates 256 reduction kernels, saving 1.57 ms/tok).
9. `handoff-2026-09-25-task2-vectorized-q4k-gemv.md` — Task 2 Vectorized Cooperative GEMV (62.06 tok/s on H100, 128-bit/64-bit vector loads).
10. `handoff-2026-09-25-h100-baseline-and-quartz-setup.md` — Fast-start cheat sheet (one-command make targets), H100 SXM5 benchmark results, 2,060-kernel decode latency breakdown, 0.5B debugging setup.

## Fast-Start Commands (Root Makefile)

```bash
# On Quartz (ssh g37) or Lair (ssh lair-g6):
cd ~/projects/agent-handoffs   # (or /u/demistry/agent-handoffs on Lair)

make info            # Print auto-detected cluster, GPU, paths, and environment settings
make test-units      # Run Q4_K, Q6_K, and nv_argmax unit sweeps (ALL OK)
make parity          # Verify 27B model logits (top-5 match: [271, 25, 11751, 248044, 198])
make token-ab        # Verify 27B greedy token sequence identity [381, 310, 5790, 421, 279, ...] (80.04 tok/s)
make bench-tg        # Measure steady-state tinygrad decode throughput (512 ctx, 20 steps: 85.31 tok/s)
make bench-llama     # Measure reference llama.cpp decode throughput (86.18 ± 1.04 tok/s)
make bench-tg-05b    # Fast-iteration test on 0.5B model (136.77 tok/s, 7.31 ms/tok)
make bench-llama-05b # Fast-iteration reference llama.cpp on 0.5B model (919 tok/s)
```

## Ground truth & Host Mappings

- **Active Compute Node**: `g37.quartz.uits.iu.edu` (`ssh g37`), NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s bandwidth). Slurm-gated (needs an active job, partition `h100-debu`, 1 h limit — `sbatch` or `srun` to renew with `-A r00117`).
- **Secondary Node**: `lair-g6` (`ssh lair-g6`), NVIDIA L40S 48GB GDDR6 (864 GB/s bandwidth).
- **tinygrad Workdir**:
  - Quartz: `$(HOME)/projects/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, HEAD `4026d67ff`).
  - Lair: `/u/demistry/tinygrad-src` (branch `qwen27b-nv-q8-kernel`).
- **Model Paths**:
  - 27B Q4_K_M (Quartz): `/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
  - 0.5B Q4_K_M (Quartz): `/N/scratch/demistry/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`
  - 27B Q4_K_M (Lair): `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`
- **Launcher**: `/N/u/demistry/Quartz/projects/agent-handoffs/Makefile` (host-aware, auto-selects cluster settings).

## Benchmark Comparison Table (Q4_K_M, ctx=512, f16 KV)

| GPU / Platform | Engine | Decode tok/s | Decode ms/tok | Parity Ratio | Logit Parity |
|---|---|---:|---:|:---:|:---:|
| **H100 SXM5 80GB** (`g37`) | **llama.cpp** (`llama-bench`, clean, re-measured 2026-09-25) | **86.18 ± 1.04** | **11.60 ms** | 100% | Reference |
| **H100 SXM5 80GB** (`g37`) | **tinygrad (`4026d67ff`: Vectorized Q4_K Scales)** | **85.31** | **11.72 ms** | **99.0%** | **Bit-exact match** |
| H100 SXM5 80GB (`g37`) | tinygrad (`cfd17abe5`: SwiGLU fused SiLU*up, nv_argmax) | 83.76 | 11.94 ms | 97.2% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (`f0d0522c1`: nv_argmax custom kernel) | 82.91 | 12.06 ms | 96.2% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (TODO 1+2: fused add_rmsnorm, compact q8, `38342a3be`) | 73.88 | 13.54 ms | 85.9% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (Actions 1–3, `c14c50207`) | 68.79 | 14.54 ms | 79.8% | Match (diff $\le 0.0019$) |
| H100 SXM5 80GB (`g38`) | tinygrad (Task 2: Vectorized Coop) | 62.06 | 16.11 ms | 72.0% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (Task 1: Heuristic r_256) | 53.07 | 18.84 ms | 61.6% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (09-24 Baseline) | 38.47 | 25.99 ms | 44.6% | 0.9978 sim |
| **L40S 48GB** (`lair-g6`) | **llama.cpp** (`llama-bench`) | **37.95 ± 0.28** | **26.35 ms** | 100% | Reference |
| L40S 48GB (`lair-g6`) | **tinygrad** (Cooperative Warp) | **31.10** | **32.15 ms** | **82.0%** | 0.9980 sim |

## Ranked Next Steps to Close the Final 0.12 ms Gap to llama.cpp (11.60 ms)

1. **Cross-Block 2nd Residual Addition Fusion (`E_40_32_4`)**:
   - In each of the 15 dense blocks, fold `out = h + ffn_out` into the next block's `attn_norm` using `nv_add_rmsnorm`.
   - Eliminates 15 `E_40_32_4` kernel launches (~0.12–0.18 ms/tok), closing the final 0.12 ms gap.
2. **Vectorize `nv_linear_q6_k_v2` Scales**:
   - Apply the same header/scale vectorization to `nv_linear_q6_k_v2` (12 calls per decode step in `tinygrad/llm/kernels/nv_q6k.py`).
3. **Double-buffering / explicit software pipelining in `nv_linear_q4_k_v4`**:
   - Overlap `ld.global.v4.u32` weight loads of iteration $i+1$ with `__dp4a` arithmetic of iteration $i$.

**Benchmark hygiene reminder:** Always ensure `llama-server` is stopped before running benchmarks (`make stop`), as resident background servers throttle H100 GPU boost clocks.
