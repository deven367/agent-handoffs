# Handoff: Optimization Actions 1, 2, 3, and TODO 1 Landed (2026-09-25)

**Hardware Target**: NVIDIA H100 SXM5 80GB HBM3 (`g37.quartz.uits.iu.edu`, Quartz cluster)  
**tinygrad Git Commit**: `231786562` (`perf(nv): fuse Q8_0 activation quantization directly into nv_rmsnorm`)  
**Prior Commit**: `16c494e99` (`perf(nv): fused QK L2 norm, 64-bit vectorized Q6_K GEMV, and intra-warp activation quantize`)  
**Branch**: `qwen27b-nv-q8-kernel` on `github.com:deven367/tinygrad.git`  
**Model**: `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (15.65 GiB)  

Prior handoffs:
- [`handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md)
- [`handoff-2026-09-25-task3-fused-rmsnorm.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/handoff-2026-09-25-task3-fused-rmsnorm.md)
- [`ACTIVE.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/ACTIVE.md)

---

## 1. Executive Summary & Accomplishments

All 3 targeted performance actions PLUS **TODO 1 (Fused RMSNorm + Q8 Quantization)** were successfully implemented, microbenchmarked, unit-tested, parity-verified, committed, and pushed:

1. **Action 1: Fused QK L2 Normalization (`nv_normalize`) for `GatedDeltaNetBlock`**:
   - Replaced graph-compiled `Tensor.normalize` at `tinygrad/llm/model.py:355` with custom fused warp reduction kernel `nv_normalize` in `tinygrad/llm/kernels/nv.py`.
   - **Completely eliminated all 128 `r_16_8` reduction kernels and all 128 `E_*` elementwise scaling kernels (256 kernels total)** from the JIT rollout graph.
   - Formula strictly preserves exact $L_2$ normalization: $x \cdot \max(\sqrt{\sum x^2}, \epsilon)^{-1}$.

2. **Action 2: 64-Bit Vectorized Q6_K GEMV (`_q6_k_v2_decode_kernel`)**:
   - Implemented `_q6_k_v2_decode_kernel` in `tinygrad/llm/kernels/nv_q6k.py` for all layers where `num_blocks % 2 == 0`.
   - 16 threads process each block pair concurrently (2 blocks per step), loading 2 consecutive words per thread with chained INT32 hardware `__dp4a` arithmetic.
   - Shared int8 scales and activation scales evaluated once across both words, halving float arithmetic and scale extractions.
   - **Isolated microbenchmark on 17408x5120 (27B down-projection layer)**:
     - Scalar `_q6_k_decode_kernel`: **11.80 ms**
     - Vectorized `_q6_k_v2_decode_kernel`: **9.61 ms** (**+22.8% speedup in isolation!**)

3. **Action 3: Single-Pass Register-Shuffle Activation Quantization (`nv_q8_quantize`)**:
   - Optimized `_q8_quantize_kernel` in `tinygrad/llm/kernels/nv.py` with intra-warp register shuffling (`__shfl_sync`).
   - Every lane loads its own activation element once and computes its quantized byte in registers. Words are packed via `_nv_shuffle_idx` without redundant DRAM round-trips.
   - **Eliminated 80% of DRAM memory loads** (from 160 scalar loads down to 32 coalesced loads per group).
   - Microbenchmark speedup: **1.19× faster** (2.28 ms $\rightarrow$ 1.91 ms across 2,000 dispatches).

4. **TODO 1 LANDED: Fused RMSNorm + Q8 Quantization (`nv_rmsnorm_q8`) (`231786562`)**:
   - Fused Q8_0 activation quantization directly into the store pass of `nv_rmsnorm`.
   - Each thread in the warp computes the group scale and quantized int8 values in registers, packing 4 bytes into a uint32 word via `__shfl_sync` without any DRAM round-trips.
   - Direct outputs `(q, scale)` are pre-populated into `_q8_cache` across the UOp DAG hierarchy (`res`, `res.uop`, `out`, `out.uop`).
   - Subsequent `q4_k_linear` and `q6_k_linear` layers consume `(q, scale)` directly from cache.
   - **Kernel Census Result**: `nv_rmsnorm_q8` now replaces `nv_rmsnorm`, and separate `nv_q8_quantize` dispatches dropped from 5 down to 3 in unique programs.

---

## 2. Verification Results

### Unit Tests (`make test-units`)
- `test_coop_q4k.py`: **ALL OK** across all sweep shapes.
- `sweep_q6k.py`: **ALL OK** across all sweep shapes (`maxabs <= 3.0e-5`, `scaled_rel <= 3.0e-7`).

### Full 27B End-to-End Numerical Parity (`make parity`)
Evaluated across all 248,320 vocabulary tokens:
```text
cs=1 argmax=271
cs=1 top5=[(271, 33.7863), (25, 21.0458), (11751, 18.5683), (198, 17.0788), (248044, 16.7757)]
cs=1 n=248320 max=+33.7863 min=-12.1791 sum=-803285.56
```
- **Bit-exact argmax: `271`** ✅
- **All Top-5 tokens match**: `[271, 25, 11751, 198, 248044]` ✅

---

## 3. Next Actionable TODOs for the Incoming Agent (Ranked by ROI)

To close the remaining ~2.0 ms gap to `llama.cpp` (11.60 ms), the following tasks are queued in prioritized order:

### TODO 1 (Completed): Fused RMSNorm + Q8 Quantization (`nv_rmsnorm_q8`)
- **Status**: **LANDED & VERIFIED** in commit `231786562`.

### TODO 2 (Next Priority, ~0.5–0.7 ms/tok): Fused Residual Addition + RMSNorm (`nv_add_rmsnorm`)
- **Location**: `tinygrad/llm/kernels/nv.py` and `tinygrad/llm/model.py:158`.
- **Root Cause**: In each of the 64 layers, the residual addition `h = x + attn_output` runs as a standalone elementwise kernel (`E_*`), writing the sum to DRAM before RMSNorm reads it back.
- **Action**:
  - Extend `nv_rmsnorm` to accept an optional residual tensor: `nv_rmsnorm(x, weight, residual=None)`.
  - In the reduction loop, each thread loads `val = x[idx] + residual[idx]` in registers, accumulates `acc += val^2`, and optionally writes the updated residual sum back if needed.
- **Expected Speedup**: Eliminates 64 kernel launches and 64 DRAM round-trips, saving **~0.5–0.7 ms/tok**.

### TODO 3 (Medium ROI, ~0.2–0.4 ms/tok): Vectorized 128-Bit Stores for `nv_q8_quantize`
- **Location**: `tinygrad/llm/kernels/nv.py:75`.
- **Action**:
  - In `_q8_quantize_kernel` and `_rmsnorm_kernel`, pack the 8 uint32 words into 2 `uint4` (128-bit) stores using lanes 0 and 1, skipping writes for lanes 2..31.
  - This avoids uncoalesced/redundant writes across the remaining 24 threads in the warp.

---

## 4. Cheat Sheet & Traps to Avoid for Incoming Agent

```bash
# Connect to compute node (H100 SXM5 80GB):
ssh g37   # (or ssh g38)

cd ~/projects/agent-handoffs

# 1. Run unit test sweeps (Q4_K + Q6_K):
make test-units
# Output must be: ALL OK

# 2. Check full 27B model numerical parity (argmax 271, top-5 match):
make parity
# Output:
# cs=1 argmax=271
# cs=1 top5=[(271, 33.7863), (25, 21.0458), (11751, 18.5683), (198, 17.0788), (248044, 16.7757)]

# 3. Benchmark steady-state decode throughput:
make bench-tg

# 4. Fast sub-second debugging iteration with 0.5B model:
make bench-tg-05b
```
