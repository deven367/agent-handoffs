# HANDOFF — Custom Two-Stage nv_argmax Landed (2026-09-25)

## 1. Executive Summary

- **Objective**: Eliminate the ~1.5 ms/tok latency bottleneck in the vocabulary argmax reduction stage (`r_2_32_4_970`) on NVIDIA H100 SXM5 80GB (`g37.quartz.uits.iu.edu`) to close the gap to `llama.cpp` (86.18 ± 1.04 tok/s, 11.60 ms/tok).
- **Outcome**: **SUCCESS**. Replaced generic reduction with a custom CUDA two-stage packed 64-bit warp argmax kernel (`nv_argmax`).
- **Throughput**: Improved steady-state decode throughput from **73.83 tok/s (13.54 ms/tok)** to **82.91 tok/s (12.06 ms/tok)**.
- **Latency Eliminated**: **-1.48 ms/tok** eliminated in a single commit (`f0d0522c1`). Total latency reduction from initial 09-24 baseline: **-13.93 ms/tok**.
- **Remaining Gap to llama.cpp**: Narrowed to **0.46 ms/tok** (**96.2% parity** with `llama.cpp` reference).
- **Parity & Correctness**: 100% bit-exact across unit tests (`make test-units`), logit parity (`make parity`), and greedy token sequences (`make token-ab`).

---

## 2. Benchmark Comparison Table (H100 SXM5 80GB, Q4_K_M, ctx=512)

| Engine / Configuration | Decode Throughput | Decode Latency | Latency vs Baseline | Remaining Gap | Parity vs llama.cpp | Logit Parity / Token Identity |
|---|---:|---:|---:|---:|:---:|:---:|
| Initial Baseline (09-24) | 38.47 tok/s | 25.99 ms/tok | 0.00 ms | 14.39 ms | 44.6% | 0.9978 cosine sim |
| Actions 1–3 Landed (`c14c50207`) | 68.79 tok/s | 14.54 ms/tok | -11.45 ms | 2.94 ms | 79.8% | Match (diff $\le 0.0019$) |
| Fused Add+RMSNorm (`38342a3be`) | 73.88 tok/s | 13.54 ms/tok | -12.45 ms | 1.94 ms | 85.9% | Bit-exact match |
| **Current (`f0d0522c1`: `nv_argmax`)** | **82.91 tok/s** | **12.06 ms/tok** | **-13.93 ms** | **0.46 ms** | **96.2%** | **Bit-exact match** |
| **Reference: `llama.cpp` (`llama-bench`)** | **86.18 ± 1.04 tok/s** | **11.60 ms/tok** | **-14.39 ms** | **0.00 ms** | **100% (Goal)** | Reference |

- **Token Probe (`make token-ab`)**: Decode speed 77.70 tok/s (vs 69.78 tok/s baseline). Sequence: `[381, 310, 5790, 421, 279, 491, 2936, 1000, 381]` (bit-exact).
- **Logit Parity (`make parity`)**: Bit-exact `argmax=271`, top-5 match `[(271, 34.0718), (25, 19.5536), (11751, 18.7959), (248044, 17.0679), (198, 16.6857)]`, sum `-800723.00`.
- **0.5B Fast Benchmark (`make bench-tg-05b`)**: 126.26 tok/s (7.92 ms/tok) vs 108.33 tok/s baseline.

---

## 3. Kernel Architecture & Implementation Details

### Root Cause of Generic Reduction Overhead
Tinygrad's generic reduction scheduler scheduled the 248,320-element vocabulary argmax onto **2 blocks × 32 threads** (`r_2_32_4_970`), forcing each thread through a sequential 970-iteration loop. This took ~1.52 ms/tok despite only processing ~1 MB of logits.

### The Two-Stage Custom Kernel (`tinygrad/llm/kernels/nv.py`)

1. **Stage 1: Partial Argmax (`nv_argmax_partial`)**:
   - Grid: `(num_tokens, 970, 1)`, Block: `(32, 1, 1)`.
   - 970 warps per token process 256 vocabulary elements per warp (8 unrolled elements per thread).
   - **Lexicographical 64-bit float packing**:
     ```c
     inline __device__ uint64_t pack_val_idx(float v, int32_t idx) {
       uint32_t u = __float_as_uint(v);
       uint32_t mask = ((int32_t)u >> 31) | 0x80000000u;
       uint32_t orderable_v = u ^ mask;
       return (((uint64_t)orderable_v) << 32) | (uint32_t)(~idx);
     }
     ```
     By XORing the sign bit and inverting the index in the lower 32 bits, a standard unsigned 64-bit maximum simultaneously finds the maximum float value and breaks ties towards the lowest index in a single clock cycle.
   - Reduction within the warp uses 5 rounds of `__shfl_xor_sync(0xffffffff, best, delta)`.
   - Thread 0 unpacks and stores `(vmax, imin)` into partial buffers.

2. **Stage 2: Final Argmax (`nv_argmax_final`)**:
   - Grid: `(num_tokens, 1, 1)`, Block: `(32, 1, 1)`.
   - A single warp iterates 31 times over the 970 partial entries, maintaining the best 64-bit packed pair.
   - Final warp shuffle reduction finds the global winner.
   - Thread 0 writes the winning `uint32_t` flat index.

3. **Integration in `Transformer.forward` (`tinygrad/llm/model.py`)**:
   - Replaced `.argmax(-1, keepdim=True)` on noisy logits with `nv_argmax(noisy)`.
   - Falls back transparently to generic `.argmax(-1, keepdim=True)` on non-CUDA / unsupported devices.

---

## 4. Key Traps Discovered & Avoided

1. **TinyJit Buffer Aliasing Trap (CRITICAL)**:
   - When returning a sliced view from custom kernels in `forward` (e.g. `out32[:, :1]`), TinyJit's memory planner aliased `out32` with the next step's input tokens buffer. During rollout, the kernel overwrote its own input tokens, causing token divergence after step 1.
   - **Fix**: Slices returned from custom kernels in `forward` MUST be made `.contiguous()` (`res = out32[:, :1].contiguous()`) to guarantee input and output buffer pointers do not alias in the JIT CUDA Graph.
2. **Format String Escaping in CStyle**:
   - Tinygrad's `cstyle.py` passes `arg` through Python's `str.format()`. All literal C/C++ braces in lambdas or statement expressions MUST be doubled (`{{` and `}}`) or Python throws `KeyError`.
3. **No Reductions Left in Decode Graph**:
   - Verification with `model.rollout_jit.captured.linear` confirms that `r_*` reduction kernels in the decode rollout graph have been completely reduced from 2 to **0**.

---

## 5. Next Optimization Roadmap (Closing the Final 0.46 ms Gap)

With `nv_argmax` landed, the remaining 0.46 ms decode gap is distributed across:
1. **Elementwise Kernel Fusion (~3.9 ms total across ~511 launches)**:
   - **SwiGLU Activation Fusing (Tested on 0.5B)**: Removing `.contiguous()` from `self.ffn_gate(x).silu().contiguous() * self.ffn_up(x)` in `FFNBlock._feed_forward` yielded a **+9.73 tok/s boost (125.41 -> 135.14 tok/s, -0.57 ms/tok)** on 0.5B with bit-exact token match. This fuses SiLU with the up-projection elementwise multiplication and eliminates intermediate DRAM roundtrips (`E_136_32_4`). Ready for 27B validation.
   - **Second Residual Add**: Restructure cross-block residual add to ride into the next block's attention norm.
2. **Multi-Warp Grid-Fused RMSNorm + Q8 Quantize**:
   - Grid-launch 160 warps where each warp handles 1 Q8 group (32 elements) to eliminate 188 separate `nv_q8_quantize` kernel launches (~1.5 ms) without triggering the single-warp unrolling register-spill trap.
3. **Quantized GEMV Memory Efficiency**:
   - Streaming efficiency optimization on `nv_linear_q4_k_v4` and `nv_linear_q6_k_v2` towards the 5.0 ms HBM floor.

---

## 6. Verification Commands

```bash
# On g37 (ssh g37, cd ~/projects/agent-handoffs)
make test-units      # Unit tests (Q4_K, Q6_K, nv_argmax sweeps) -> 100% OK
make parity          # Logit parity -> argmax 271, sum -800723.00, top-5 match
make token-ab        # Token identity probe -> [381, 310, 5790, 421, 279, 491, 2936, 1000, 381]
make bench-tg        # Tinygrad steady-state decode -> 82.91 tok/s (12.06 ms/tok)
make bench-llama     # Reference llama.cpp decode -> 86.18 ± 1.04 tok/s (11.60 ms/tok)
make bench-tg-05b    # Fast 0.5B benchmark -> 126.26 tok/s (7.92 ms/tok)
```
