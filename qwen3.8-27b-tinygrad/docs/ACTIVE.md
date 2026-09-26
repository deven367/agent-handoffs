# ACTIVE — Current state and next steps

> Snapshot: 2026-09-25. Evaluated on NVIDIA H100 SXM5 80GB (`g37` on Quartz). Actions 1–3 landed (`c14c50207`) and **TODO 1–2 landed (`38342a3be`)**: Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize, **Fused Residual Add + RMSNorm (`nv_add_rmsnorm`, −64 E_* launches), Compact Q8 Activation Layout (36 B/group instead of 256 B)**. Parity verified bit-exact across all 248,320 vocabulary tokens. Unit tests 100% passing (`make test-units`). **Lever A (compact GEMV output buffers) was implemented, broke parity, and was REJECTED/reverted — the tinygrad tree is clean at `38342a3be`.**

## Read first

1. `handoff-2026-09-25-lever-a-rejected-and-reprofile.md` — **START HERE: Lever A rejected (PTX root cause; the 572 copy kernels it targeted do not exist), fresh kernel census, re-ranked next steps.**
2. `handoff-2026-09-25-fused-add-rmsnorm-and-compact-q8.md` — TODO 1+2 Landed (`38342a3be`): fused add+RMSNorm, compact Q8 layout, 73.69 tok/s, clean A/B re-measure, benchmark-hygiene finding, GEMV output-buffer lever.
3. `handoff-2026-09-25-actions-1-2-3-completed.md` — Actions 1, 2, and 3 Landed (Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize), Current Gap Analysis, and Unrolling Trap Takeaways (`c14c50207`).
4. `handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md` — GatedDeltaNet Normalization Profiling, GPU Clock Dynamics, and Next Optimization Roadmap.
5. `handoff-2026-09-25-task3-fused-rmsnorm.md` — Task 3 Single-Pass Block-Fused RMSNorm (eliminates 256 reduction kernels, saving 1.57 ms/tok).
6. `handoff-2026-09-25-task2-vectorized-q4k-gemv.md` — Task 2 Vectorized Cooperative GEMV (62.06 tok/s on H100, 128-bit/64-bit vector loads).
7. `handoff-2026-09-25-h100-baseline-and-quartz-setup.md` — Fast-start cheat sheet (one-command make targets), H100 SXM5 benchmark results, 2,060-kernel decode latency breakdown, 0.5B debugging setup.

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
| **H100 SXM5 80GB** (`g37`) | **tinygrad (TODO 1+2: fused add_rmsnorm, compact q8, `38342a3be`)** | **73.83** | **13.54 ms** | **86.0%** | **Bit-exact match** |
| H100 SXM5 80GB (`g37`) | tinygrad (Actions 1–3, `c14c50207`) | 68.79 | 14.54 ms | 79.8% | Match (diff $\le 0.0019$) |
| H100 SXM5 80GB (`g38`) | tinygrad (Task 2: Vectorized Coop) | 62.06 | 16.11 ms | 72.0% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (Task 1: Heuristic r_256) | 53.07 | 18.84 ms | 61.6% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (09-24 Baseline) | 38.47 | 25.99 ms | 44.6% | 0.9978 sim |
| **L40S 48GB** (`lair-g6`) | **llama.cpp** (`llama-bench`) | **37.95 ± 0.28** | **26.35 ms** | 100% | Reference |
| L40S 48GB (`lair-g6`) | **tinygrad** (Cooperative Warp) | **31.10** | **32.15 ms** | **82.0%** | 0.9980 sim |

## Latency Breakdown & Current Status on H100 (`38342a3be`)

Decode graph = **1,588 kernels/step**; ~13.5 ms/tok wall. Ranked by GPU time
(`JIT=0 DEBUG=2` census, sum of `tm` over the final 1,588 launches — ranking only, DEBUG=2
syncs per kernel so absolutes run ~1.3-1.5x high; full table in the 09-25 rejection handoff):

```text
├── 1. Quantized GEMV (~7.9 ms)   16.8 GB/token streamed; 5.0 ms is the HBM floor
│   ├── nv_linear_q4_k_v4 (318 calls/window)
│   └── nv_linear_q6_k_v2 (43 calls/window)
├── 2. r_2_32_4_970 (~1.5 ms, 2 calls, ~760 us each)  UNIDENTIFIED — vocab-sized
│                                                          tensor (248320 = 2*32*4*970),
│                                                          last kernels of the step →
│                                                          logits argmax/sample stage
├── 3. Activation quantization (nv_q8_quantize, 188 calls): ~1.5 ms
├── 4. Norms (nv_rmsnorm 106, nv_add_rmsnorm 47, nv_normalize 69 calls): ~2.2 ms
├── 5. Elementwise (E_40_32_4, E_64_32_3, E_136_32_4, E_16_32_4, E_3_4_4, E_16_32_4_3): ~3.9 ms
└── 6. Attention/recurrence (gated_delta_prefill, flash_decode_partial): ~0.7 ms
```

**There are zero `copy` kernels in the decode step** — the GEMV consumers read `[..., 0]`
as fused strided index arithmetic. Any lever premised on eliminating copies is dead on arrival.

## Ranked Next Steps to Close the Remaining ~1.9 ms Gap to llama.cpp (11.65 ms)

1. **The vocab argmax/sample stage — ~1.5 ms/token (11% of the step). IDENTIFIED, needs a custom
   kernel.** `r_2_32_4_970` runs the 248,320-entry argmax on **2 blocks × 32 threads** with the
   Gumbel correction fused in. Two naive fixes were measured and reverted: a nested
   `argmax(argmax)` is silently WRONG (returns a row/column, not a vocab id — its 77.62 tok/s was
   a fake win), and a provably correct `argmax`+`max`+`gather` restaging is 1.65 ms SLOWER
   (62.58 tok/s). What is needed: a custom greedy-argmax kernel in the `tinygrad/llm/kernels/`
   `nv.py` style — warp-per-32-chunk `(value,index)` partials + one single-block second stage;
   the data is ~1 MB + ~31 KB, so tens of microseconds, i.e. ~70% of the remaining gap.
2. **Fuse the remaining elementwise families** (~3.9 ms/window): SwiGLU + RoPE + the 2nd residual
   add are still separate kernels; ride the 2nd residual add into the next block's `attn_norm`
   (cross-block `Transformer.forward` restructure).
3. **GEMV streaming efficiency**: ~7.9 ms against a 5.0 ms HBM floor — the rest of the gap, but
   deep work; only after 1 and 2.
4. Multi-warp grid-fused RMSNorm + Q8 quantize (old TODO 3) stays queued behind 1-3.
5. **Kernel rules learned 2026-09-25:** (a) a custom-kernel store may only be written through an
   address that depends on the lane axis — a lane-invariant store makes the CUDA renderer gate
   the store and re-emit the last reduce butterfly, doubling the stored value; (b) the generic
   reduce scheduler will pick a 2-block launch for a 248 k-element argmax, so a hot reduce needs
   a custom kernel, not a reshape.
6. **Slurm**: partition is `h100-debug` (squeue truncates), submissions need `-A r00117`, 1 h is
   the hard max; `scontrol requeue`/`update` do not work on interactive jobs.

**Benchmark hygiene:** a resident `llama-server` (44% SM bursts) cut llama-bench 85.87 → 48.51 tok/s. Always `make stop` before benchmarking, `make serve` after.
