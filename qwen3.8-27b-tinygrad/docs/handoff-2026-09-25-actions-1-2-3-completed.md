# Handoff: Optimization Actions 1, 2, and 3 Landed (2026-09-25)

**Hardware Target**: NVIDIA H100 SXM5 80GB HBM3 (`g37.quartz.uits.iu.edu`, Quartz cluster)  
**tinygrad Git Commit**: `16c494e99` (`perf(nv): fused QK L2 norm, 64-bit vectorized Q6_K GEMV, and intra-warp activation quantize`)  
**Branch**: `qwen27b-nv-q8-kernel` on `github.com:deven367/tinygrad.git`  
**Model**: `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (15.65 GiB)  

Prior handoffs:
- [`handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/handoff-2026-09-25-gated-deltanet-normalize-and-next-steps.md)
- [`handoff-2026-09-25-task3-fused-rmsnorm.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/handoff-2026-09-25-task3-fused-rmsnorm.md)
- [`ACTIVE.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/ACTIVE.md)

---

## 1. Executive Summary & Accomplishments

All 3 targeted performance actions requested by the user were implemented, micro-benchmarked, unit-tested, parity-verified, and committed:

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

---

## 2. Verification Results

### Unit Tests (`make test-units`)
- `test_coop_q4k.py`: **ALL OK** across all sweep shapes.
- `sweep_q6k.py`: **ALL OK** across all sweep shapes (`maxabs <= 3.0e-5`, `scaled_rel <= 3.0e-7`).

### Full 27B End-to-End Numerical Parity (`make parity`)
Evaluated across all 248,320 vocabulary tokens:
```text
cs=1 argmax=271
cs=1 top5=[(271, 34.0718), (25, 19.5536), (11751, 18.7959), (248044, 17.0679), (198, 16.6857)]
cs=1 n=248320 max=+34.0718 min=-12.4540 sum=-800723.00
```
- **Bit-exact argmax: `271`** ✅
- **All Top-5 tokens match**: `[271, 25, 11751, 248044, 198]` ✅

---

## 3. Git Commit Details

- **Commit**: `16c494e99`
- **Branch**: `origin/qwen27b-nv-q8-kernel`
- **Files Modified**:
  - `tinygrad/llm/kernels/nv.py`: added `_l2norm_kernel`, `nv_normalize`, `_nv_shuffle_idx`, and shuffle-packed `_q8_quantize_kernel`.
  - `tinygrad/llm/kernels/nv_q6k.py`: added `_q6_k_v2_decode_kernel` and auto-dispatch logic.
  - `tinygrad/llm/model.py`: dispatched GatedDeltaNet QK normalization to `nv_normalize`.
