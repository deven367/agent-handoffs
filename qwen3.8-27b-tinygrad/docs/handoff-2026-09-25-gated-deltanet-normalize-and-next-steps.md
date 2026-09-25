# Handoff — GatedDeltaNet Normalization Profiling, GPU Clock Dynamics & Optimization Roadmap (2026-09-25)

Prior handoffs:
- [`handoff-2026-09-25-task3-fused-rmsnorm.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/handoff-2026-09-25-task3-fused-rmsnorm.md) — Task 3 Single-Pass Block-Fused RMSNorm (`bd17e6e1c`).
- [`handoff-2026-09-25-task2-vectorized-q4k-gemv.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/handoff-2026-09-25-task2-vectorized-q4k-gemv.md) — Task 2 128-bit/64-bit Vectorized Cooperative Warp GEMV (`c162d326b`).
- [`ACTIVE.md`](file:///Users/deven367/projects/agent-handoffs/qwen3.8-27b-tinygrad/docs/ACTIVE.md)

---

## 1. Quick-Start Cheat Sheet for Next Agent (NO EXPLORATION NEEDED)

Everything is pre-configured and automated in the root [`Makefile`](file:///Users/deven367/projects/agent-handoffs/Makefile).
Both clusters (Quartz and Lair) are fully supported with automatic cluster, GPU, and path detection.

### One-Command Operations
```bash
# Connect to active compute node (H100 SXM5 80GB):
ssh g38

# Navigate to workspace:
cd ~/projects/agent-handoffs

# Run all unit tests (Q4_K + Q6_K sweeps):
make test-units
# Output: ALL OK

# Verify numerical parity on full 27B model (bit-exact top-5 match):
make parity
# Output:
# cs=1 argmax=271
# cs=1 top5=[(271, 33.4181), (25, 20.4974), (11751, 18.8221), (248044, 17.2059), (198, 17.1326)]
# cs=1 n=248320 max=+33.4181 min=-12.2935 sum=-795259.06

# Measure steady-state tinygrad decode throughput (512 ctx, 20 steps):
make bench-tg
# Decode throughput: 62.5–68.8 tok/s depending on clock thermal state (14.5–16.0 ms/tok)

# Measure fast 0.5B debugging model (sub-second iteration):
make bench-tg-05b
# Output: ~108 tok/s (9.2 ms/tok)
```

---

## 2. Current State & What Was Investigated

### Ground Truth State
- **Active Node**: `g38.quartz.uits.iu.edu` (`ssh g38`, NVIDIA H100 SXM5 80GB HBM3, 3,350 GB/s bandwidth).
- **tinygrad Workdir**: `~/projects/tinygrad-src` on `g38` (branch `qwen27b-nv-q8-kernel`, HEAD `bd17e6e1c`).
- **All Unit Tests**: 100% PASSING (`make test-units`).
- **Parity**: Verified on 248,320 vocabulary tokens. Argmax is token `271`. Top-5 tokens: `[271, 25, 11751, 248044, 198]`.

### Key Findings This Session

1. **Rollout JIT Kernel Census (Qwen3.8-27B, 1,681 kernels)**:
   Analysis of `model.rollout_jit.captured.linear.src` revealed the exact breakdown of the remaining 14.5–16.0 ms/tok:
   ```text
   Total calls in 27B rollout_jit: 1,681
   ├── GEMV Kernels (572 calls): ~7.7 ms
   │   ├── nv_linear_q4_k_v4 (444 calls, 128-bit vectorized): ~5.2 ms
   │   ├── nv_linear_q4_k_v2 (64 calls, 64-bit vectorized): ~0.9 ms
   │   └── nv_linear_q6_k (64 calls, 32-bit cooperative): ~1.6 ms  <-- Target 2
   └── Non-GEMV Kernels (1,109 calls): ~6.9–8.3 ms
       ├── nv_rmsnorm (193 calls, fused single-pass): ~2.0 ms
       ├── nv_q8_quantize (192 calls, activation quantization): ~1.8 ms  <-- Target 3
       ├── r_16_8 (128 calls, QK L2 reduction in GatedDeltaNet): ~0.7 ms  <-- Target 1
       ├── E_* elementwise scaling for QK normalization (128 calls): ~0.6 ms  <-- Target 1
       └── Residual adds, SwiGLU, and activations (~467 calls): ~1.8 ms
   ```

2. **The GatedDeltaNet `r_16_8` Reductions (256 kernels total)**:
   In `tinygrad/llm/model.py:355`:
   ```python
   q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
           .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
   ```
   For Qwen3.8-27B:
   - `head_k_dim = 128`
   - `num_k_heads = 16`
   - 64 layers $\times$ 2 (`q` and `k`) = **128 separate reduction kernels (`r_16_8`)** plus **128 elementwise scaling kernels (`E_*`)**, total **256 kernels**!
   - Math: `Tensor.normalize` is L2 norm: $x \cdot \frac{1}{\max(\sqrt{\sum x^2}, \epsilon)}$.
   - Standalone microbenchmark confirmed that a block-fused warp reduction `nv_normalize` matches `Tensor.normalize` with max difference $\le 2.4 \times 10^{-4}$ (within float16 rounding).

3. **GPU Idle Clocks vs Boost Dynamics on H100**:
   - `nvidia-smi -q -d CLOCK` revealed that when idle, H100 SXM5 drops SM clocks to **345 MHz** (vs 1980 MHz max boost).
   - In short 20-token runs (~0.3s runtime), if the GPU is coming out of idle, clock ramp-up can produce 62.5 tok/s, whereas when warm or running continuous generation, throughput reaches ~68.8 tok/s.

---

## 3. High-Priority Next Actions to Close the 2.94 ms Gap to llama.cpp (11.60 ms)

### Action 1: Fused QK `nv_normalize` for `GatedDeltaNetBlock` (Estimated savings: ~1.0–1.3 ms/tok)
- **Target**: Replace `q.normalize(dim=-1, eps=qk_eps)` and `k.normalize(dim=-1, eps=qk_eps)` in `model.py:355`.
- **Implementation**:
  Add `nv_normalize(x, eps)` to `tinygrad/llm/kernels/nv.py`:
  ```python
  @functools.cache
  def _l2norm_kernel(out:UOp, x:UOp, tokens:int|UOp, dim:int, eps:float) -> UOp:
    token = UOp.range(tokens, 0, AxisType.GLOBAL)
    lane = UOp.range(32, 1, AxisType.LOCAL)
    elems = dim // 32
    acc = UOp.const(0, dtypes.float32)
    for i in range(elems):
      idx = lane + i * 32
      v = x[token, idx].float()
      acc = acc + v * v
    total = _warp_reduce(acc)
    # L2 norm formula: 1.0 / max(sqrt(sum), eps)
    norm = (total.sqrt().maximum(eps)).reciprocal()
    stores = []
    for i in range(elems):
      idx = lane + i * 32
      v = x[token, idx].float()
      stores.append(out[token, idx].store((v * norm).cast(out.dtype)))
    return UOp.group(*stores).end(token, lane).sink(arg=KernelInfo(name="nv_l2norm", opts_to_apply=()))
  ```
- **Validation**: Check `make test-units` and `make parity`. Top-5 tokens must match `[271, 25, 11751, 248044, 198]`.

### Action 2: Vectorized Q6_K GEMV (`nv_q6k.py`) (Estimated savings: ~0.4–0.6 ms/tok)
- **Target**: 64 down-projection / MoE layers use `nv_linear_q6_k` (~1.6 ms latency) with 32-bit scalar loads.
- **Implementation**: Adapt the Task 2 vectorized pattern to `nv_q6k.py` using 64-bit vector loads (`uint2`).

### Action 3: Activation Quantization Optimization (`nv_q8_quantize`) (Estimated savings: ~0.4–0.5 ms/tok)
- **Target**: 192 calls taking ~1.8 ms. Vectorize the max-abs reduction and store operations.
