# Handoff: Vectorized 128-bit Q4_K Scales & GEMV Optimization Landed

**Date**: 2026-09-25 (End-of-day / Bedtime Handoff)  
**Target**: Close remaining gap to `llama.cpp` on NVIDIA H100 SXM5 80GB (`g37.quartz.uits.iu.edu`)  
**Tinygrad Commit**: `4026d67ff` on branch `qwen27b-nv-q8-kernel`  
**Result**: Steady-state decode throughput increased from 83.76 tok/s (11.94 ms/tok) to **85.31 tok/s (11.72 ms/tok)**, closing **99.0%** of the gap to `llama.cpp` reference (**86.18 ± 1.04 tok/s, 11.60 ms/tok**). The remaining gap is now just **0.12 ms/tok**!

---

## 1. Summary of Results

| Configuration / Stage | Decode tok/s | Decode ms/tok | vs Baseline | Remaining Gap | Parity vs llama.cpp | Parity Status |
|---|---:|---:|---:|---:|:---:|:---:|
| Initial Baseline (09-24) | 38.47 tok/s | 25.99 ms | 0.00 ms | 14.39 ms | 44.6% | 0.9978 cosine sim |
| Actions 1–3 Landed (`c14c50207`) | 68.79 tok/s | 14.54 ms | -11.45 ms | 2.94 ms | 79.8% | Match (diff $\le 0.0019$) |
| Fused Add+RMSNorm (`38342a3be`) | 73.88 tok/s | 13.54 ms | -12.45 ms | 1.94 ms | 85.9% | Bit-exact match |
| `nv_argmax` Landed (`f0d0522c1`) | 82.91 tok/s | 12.06 ms | -13.93 ms | 0.46 ms | 96.2% | Bit-exact match |
| SwiGLU Fused (`cfd17abe5`) | 83.76 tok/s | 11.94 ms | -14.05 ms | 0.34 ms | 97.2% | Bit-exact match |
| **Current (`4026d67ff`: Vectorized Q4_K Scales)** | **85.31 tok/s** | **11.72 ms** | **-14.27 ms** | **0.12 ms** | **99.0%** | **Bit-exact match** |
| **Reference: `llama.cpp` (`llama-bench`)** | **86.18 ± 1.04 tok/s** | **11.60 ms** | **-14.39 ms** | **0.00 ms** | **100% (Goal)** | Reference |

- **Two-run confirmation on 27B**: Run 1 = **85.31 tok/s (11.72 ms)**, Run 2 = **85.15 tok/s (11.74 ms)**. Average = **85.23 tok/s (11.73 ms)**.
- **Inside 1-sigma uncertainty window**: `llama.cpp` is $86.18 \pm 1.04$ tok/s ($[85.14, 87.22]$ tok/s). Tinygrad at **85.31 tok/s** is now officially within the 1-sigma performance band of `llama.cpp`!
- **Greedy Token Probe (`make token-ab`)**: Speed improved from 78.79 tok/s to **80.04 tok/s (12.49 ms/tok)** with bit-exact sequence identity: `[381, 310, 5790, 421, 279, 491, 2936, 1000, 381]`.
- **Logit Parity (`make parity`)**: Bit-exact `argmax=271`, top-5 match `[(271, 34.0718), (25, 19.5536), (11751, 18.7959), (248044, 17.0679), (198, 16.6857)]`, sum `-800723.00`.
- **Unit Sweeps (`make test-units`)**: 100% passing across Q4_K, Q6_K, and `nv_argmax`.
- **0.5B Model Test (`make bench-tg-05b`)**: 136.77 tok/s (7.31 ms/tok) vs 135.14 tok/s.

---

## 2. Technical Detail: Vectorized Q4_K Scale Loading & In-Register Unpack

### The Problem
In `tinygrad/llm/kernels/nv_q4k.py`, each loop iteration loaded `sc_even`, `m_even`, `sc_odd`, and `m_odd` via `_load_byte(raw, base, offset)`.
Because `offset` was conditionally computed via `(subgroup < 4).where(...)`, tinygrad's compiler was unable to vectorize the scale loads. This generated **6 to 8 separate 32-bit scalar loads (`ld.global.u32`)** per block iteration.
Across 20 blocks in attention projections and 68 blocks in `ffn_down`, each thread issued up to 136 unvectorized scalar loads, causing cache line thrashing and register replay stalls.

### The Solution
In GGML Q4_K, each 256-weight block contains 144 bytes = 36 uint32 words:
- Word 0: `d` (16-bit fp16) | `dmin` (16-bit fp16)
- Word 1: `scales[0..3]` (4 bytes)
- Word 2: `scales[4..7]` (4 bytes)
- Word 3: `scales[8..11]` (4 bytes)
- Words 4..35: `qs[0..127]` (32 words, 4-bit quantized weights)

Because `base = (output * num_blocks + b) * 36`, `base` is always divisible by 4 (`36 % 4 == 0`), making `raw[base]` 16-byte aligned.

We replaced the scattered scalar loads with a single contiguous 128-bit load:
```python
# Vectorized 128-bit header load (d, dmin, scales 0..11)
hdr0 = raw[base]
hdr1 = raw[base + 1]
hdr2 = raw[base + 2]
hdr3 = raw[base + 3]
```
Tinygrad compiles `hdr0..hdr3` into a single 128-bit vectorized load:
```cuda
uint4 val2 = (*((uint4*)((data1_589824+(alu1+144)))));
```
All 8 threads in the cooperative block group access the exact same 16 bytes at `base`, allowing the GPU memory subsystem to coalesce into a single broadcast cache transaction.

We then derived closed-form in-register ALU bit-shifts and masks:
```python
def _q4k_scales(hdr0:UOp, hdr1:UOp, hdr2:UOp, hdr3:UOp, pair:UOp) -> tuple[UOp, UOp, UOp, UOp, UOp, UOp]:
  shift_even = ((pair & 1) * 16).cast(dtypes.uint32)
  shift_odd  = shift_even + 8
  sc_even = (pair < 2).where((hdr1 >> shift_even) & 63, ((hdr3 >> shift_even) & 15) | ((((hdr1 >> shift_even) & 255) >> 6) << 4))
  m_even  = (pair < 2).where((hdr2 >> shift_even) & 63, (((hdr3 >> shift_even) & 255) >> 4) | ((((hdr2 >> shift_even) & 255) >> 6) << 4))
  sc_odd  = (pair < 2).where((hdr1 >> shift_odd) & 63, ((hdr3 >> shift_odd) & 15) | ((((hdr1 >> shift_odd) & 255) >> 6) << 4))
  m_odd   = (pair < 2).where((hdr2 >> shift_odd) & 63, (((hdr3 >> shift_odd) & 255) >> 4) | ((((hdr2 >> shift_odd) & 255) >> 6) << 4))
  d = _half(hdr0 & 0xffff)
  dmin = _half((hdr0 >> 16).cast(dtypes.uint32) & 0xffff)
  return d, dmin, sc_even, m_even, sc_odd, m_odd
```
This was applied identically to `_q4_k_v4_decode_kernel`, `_q4_k_v2_decode_kernel`, and `_q4_k_decode_kernel`.

**Verification**:
- Pre-tested over 100,000 random bit patterns against the reference unpack logic — bit-exact match.
- Code generated: Zero scalar loads from weights buffer `data1`. Every memory load is now 128-bit (`uint4`).

---

## 3. Environment & Repositories

- **Cluster Node**: `g37.quartz.uits.iu.edu` (`ssh g37`), NVIDIA H100 SXM5 80GB HBM3.
- **Git Repositories**:
  - `tinygrad-src` on `g37`: Branch `qwen27b-nv-q8-kernel`, HEAD at `4026d67ff` (committed and pushed to origin).
  - `agent-handoffs`: Branch `main`.
- **Verification Commands** (run from `~/projects/agent-handoffs`):
  ```bash
  make test-units      # All unit tests pass
  make parity          # Bit-exact logits (argmax=271, sum=-800723.00)
  make token-ab        # Bit-exact greedy sequence, 80.04 tok/s
  make bench-tg        # 85.31 tok/s (11.72 ms/tok)
  make bench-tg-05b    # 136.77 tok/s (7.31 ms/tok)
  ```

---

## 4. Next Agent Action Items (To Close the Final 0.12 ms)

The remaining gap to llama.cpp is only **0.12 ms/tok** (11.72 ms vs 11.60 ms). Two concrete levers can easily close this final gap:

1. **Lever 1: Cross-Block 2nd Residual Addition Fusion (`E_40_32_4`)**:
   - In each of the 15 dense blocks, fold `out = h + ffn_out` into the subsequent block's `attn_norm` using `nv_add_rmsnorm`.
   - Eliminates 15 `E_40_32_4` kernel launches (~0.12–0.18 ms/tok), which alone should push tinygrad past 86.18 tok/s.
2. **Lever 2: Vectorize `nv_linear_q6_k_v2` Scales**:
   - Apply the same header/scale vectorization to `nv_linear_q6_k_v2` (12 calls per decode step in `tinygrad/llm/kernels/nv_q6k.py`).
