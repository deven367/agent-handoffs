# ACTIVE — Current state and next steps

> Snapshot: 2026-09-25. Evaluated on NVIDIA H100 SXM5 80GB (`g37` on Quartz). Actions 1–3 landed (`c14c50207`), TODO 1–2 landed (`38342a3be`), Custom Two-Stage Packed 64-bit Argmax landed (`f0d0522c1`), and **SwiGLU SiLU*up Fusion Landed (`cfd17abe5`)**: Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize, Fused Residual Add + RMSNorm (`nv_add_rmsnorm`), Compact Q8 Activation Layout, Two-Stage Custom Greedy Argmax (`nv_argmax`), and **Fused SwiGLU SiLU*up (removed `.contiguous()` barrier in `FFNBlock._feed_forward`)**. Decode speed broke the 12 ms barrier, reaching **83.76 tok/s (11.94 ms/tok)**, closing the gap to llama.cpp to **0.34 ms/tok (97.2% parity)**. Parity verified bit-exact across all 248,320 vocabulary tokens (`make parity`), and greedy token rollout verified identical (`make token-ab`). Unit tests 100% passing (`make test-units`).

## Read first

1. `handoff-2026-09-25-nv-argmax-landed.md` — **START HERE: custom two-stage nv_argmax kernel implementation, SwiGLU SiLU*up fusion, benchmark results (83.76 tok/s), and current 0.34 ms gap.**
2. `handoff-2026-09-25-next-agent-argmax-kernel.md` — prior handoff proposing the argmax kernel and analyzing the 1.5 ms bottleneck.
3. `handoff-2026-09-25-lever-a-rejected-and-reprofile.md` — evidence: Lever A rejected (PTX root cause, zero copy kernels), fresh kernel census, and the two failed argmax restagings.
4. `handoff-2026-09-25-fused-add-rmsnorm-and-compact-q8.md` — TODO 1+2 Landed (`38342a3be`): fused add+RMSNorm, compact Q8 layout, 73.69 tok/s, clean A/B re-measure, benchmark-hygiene finding, GEMV output-buffer lever.
5. `handoff-2026-09-25-actions-1-2-3-completed.md` — Actions 1, 2, and 3 Landed (Fused QK L2 Norm, 64-bit Vectorized Q6_K, Intra-Warp Q8 Quantize), Current Gap Analysis, and Unrolling Trap Takeaways (`c14c50207`).
6. `handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md` — GatedDeltaNet Normalization Profiling, GPU Clock Dynamics, and Next Optimization Roadmap.
7. `handoff-2026-09-25-task3-fused-rmsnorm.md` — Task 3 Single-Pass Block-Fused RMSNorm (eliminates 256 reduction kernels, saving 1.57 ms/tok).
8. `handoff-2026-09-25-task2-vectorized-q4k-gemv.md` — Task 2 Vectorized Cooperative GEMV (62.06 tok/s on H100, 128-bit/64-bit vector loads).
9. `handoff-2026-09-25-h100-baseline-and-quartz-setup.md` — Fast-start cheat sheet (one-command make targets), H100 SXM5 benchmark results, 2,060-kernel decode latency breakdown, 0.5B debugging setup.

## Fast-Start Commands (Root Makefile)

```bash
# On Quartz (ssh g37) or Lair (ssh lair-g6):
cd ~/projects/agent-handoffs   # (or /u/demistry/agent-handoffs on Lair)

make info            # Print auto-detected cluster, GPU, paths, and environment settings
make test-units      # Run Q4_K, Q6_K, and nv_argmax unit sweeps (ALL OK)
make parity          # Verify 27B model logits (top-5 match: [271, 25, 11751, 248044, 198])
make token-ab        # Verify 27B greedy token sequence identity [381, 310, 5790, 421, 279, ...]
make bench-tg        # Measure steady-state tinygrad decode throughput (512 ctx, 20 steps: 83.76 tok/s)
make bench-llama     # Measure reference llama.cpp decode throughput (85.87-86.18 tok/s clean)
make bench-tg-05b    # Fast-iteration test on 0.5B model (135.14 tok/s, 7.40 ms/tok)
make bench-llama-05b # Fast-iteration reference llama.cpp on 0.5B model (919 tok/s)
```

## Ground truth & Host Mappings

- **Active Compute Node**: `g37.quartz.uits.iu.edu` (`ssh g37`), NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s bandwidth). Slurm-gated (needs an active job, partition `h100-debu`, 1 h limit — `scontrol requeue` to renew).
- **Secondary Node**: `lair-g6` (`ssh lair-g6`), NVIDIA L40S 48GB GDDR6 (864 GB/s bandwidth).
- **tinygrad Workdir**:
  - Quartz: `$(HOME)/projects/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, HEAD `cfd17abe5`).
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
| **H100 SXM5 80GB** (`g37`) | **tinygrad (`cfd17abe5`: SwiGLU fused SiLU*up, nv_argmax)** | **83.76** | **11.94 ms** | **97.2%** | **Bit-exact match** |
| H100 SXM5 80GB (`g37`) | tinygrad (`f0d0522c1`: nv_argmax custom kernel) | 82.91 | 12.06 ms | 96.2% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (TODO 1+2: fused add_rmsnorm, compact q8, `38342a3be`) | 73.88 | 13.54 ms | 85.9% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (Actions 1–3, `c14c50207`) | 68.79 | 14.54 ms | 79.8% | Match (diff $\le 0.0019$) |
| H100 SXM5 80GB (`g38`) | tinygrad (Task 2: Vectorized Coop) | 62.06 | 16.11 ms | 72.0% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (Task 1: Heuristic r_256) | 53.07 | 18.84 ms | 61.6% | Bit-exact match |
| H100 SXM5 80GB (`g37`) | tinygrad (09-24 Baseline) | 38.47 | 25.99 ms | 44.6% | 0.9978 sim |
| **L40S 48GB** (`lair-g6`) | **llama.cpp** (`llama-bench`) | **37.95 ± 0.28** | **26.35 ms** | 100% | Reference |
| L40S 48GB (`lair-g6`) | **tinygrad** (Cooperative Warp) | **31.10** | **32.15 ms** | **82.0%** | 0.9980 sim |

## Latency Breakdown & Current Status on H100 (`f0d0522c1`)

Decode graph = **1,586 kernels/step**; ~12.06 ms/tok wall. Ranked by GPU time
(`JIT=0 DEBUG=2` census, sum of `tm` over launches):

```text
├── 1. Quantized GEMV (~7.9 ms)   16.8 GB/token streamed; 5.0 ms is the HBM floor
│   ├── nv_linear_q4_k_v4 (318 calls/window)
│   └── nv_linear_q6_k_v2 (43 calls/window)
├── 2. Elementwise (E_40_32_4, E_64_32_3, E_136_32_4, etc.): ~3.9 ms
├── 3. Norms (nv_rmsnorm 106, nv_add_rmsnorm 47, nv_normalize 69 calls): ~2.2 ms
├── 4. Activation quantization (nv_q8_quantize, 188 calls): ~1.5 ms
├── 5. Attention/recurrence (gated_delta_prefill, flash_decode_partial): ~0.7 ms
└── 6. Vocab Argmax (nv_argmax_partial + nv_argmax_final): ~0.04 ms (ELIMINATED 1.48 ms overhead!)
```

**There are zero `copy` kernels and zero generic `r_*` reduction kernels in the decode rollout graph.**
`r_2_32_4_970` was completely eliminated by `nv_argmax`.

## Ranked Next Steps to Close the Final 0.46 ms Gap to llama.cpp (11.60 ms)

1. **Fuse the remaining elementwise families** (~3.9 ms/window, ~511 small kernels):
   - SwiGLU activation (`x * silu(gate)`) into a dedicated warp kernel.
   - 2nd residual addition ride into the next block's `attn_norm` (cross-block `Transformer.forward` restructure).
2. **Multi-warp grid-fused RMSNorm + Q8 quantize**:
   - Grid-launch 160 warps where each warp handles 1 Q8 group (32 elements) to eliminate 188 separate `nv_q8_quantize` kernel launches (~1.5 ms) without triggering the single-warp unrolling register-spill trap.
3. **GEMV streaming efficiency**:
   - ~7.9 ms against a 5.0 ms HBM floor — the rest of the gap, but deep work; only after 1 and 2.
4. **Kernel rules learned 2026-09-25:**
   - (a) Slices returned from custom kernels in `forward` MUST be `.contiguous()` (`res = out32[:, :1].contiguous()`) to prevent TinyJit CUDA Graph buffer aliasing across rollout steps.
   - (b) A custom-kernel store may only be written through an address that depends on the lane axis — lane-invariant stores cause re-emitted butterfly shuffles.
   - (c) String format escaping in `cstyle.py`: literal C++ braces `{{` and `}}` must be doubled.
5. **Slurm**: partition is `h100-debug` (squeue truncates), submissions need `-A r00117`, 1 h is the hard max; `scontrol requeue`/`update` do not work on interactive jobs.

**Benchmark hygiene:** a resident `llama-server` (44% SM bursts) cut llama-bench 85.87 → 48.51 tok/s. Always `make stop` before benchmarking, `make serve` after.
