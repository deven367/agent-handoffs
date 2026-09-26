# Engineering Blueprint: Bringing tinygrad on Par with llama.cpp for Mimo-Qwen

**Target Model**: `mimo-qwen-q8_0.gguf` (Qwen 3.5 9B hybrid architecture, 8.86 GiB, 32 transformer layers).  
**Current Baseline (NVIDIA H100 SXM5 80GB)**:
- `llama.cpp`: **210.29 ± 2.26 tok/s (4.75 ms/tok)**
- `tinygrad`: **162.80 tok/s (6.14 ms/tok)**
- **Net Gap**: **1.39 ms/tok** (llama.cpp is 1.29x faster)

**Current Baseline (NVIDIA L40S 48GB)**:
- `llama.cpp`: **73.57 ± 0.44 tok/s (13.59 ms/tok)**
- `tinygrad`: **60.11 tok/s (16.64 ms/tok)**
- **Net Gap**: **3.05 ms/tok** (llama.cpp is 1.22x faster)

---

## 1. Where Does the 1.39 ms / 3.05 ms Gap Come From?

A single decode step in `mimo-qwen` executes:
- 32 transformer blocks (24 GatedDeltaNet linear recurrent layers + 8 full attention layers).
- 427 Q8_0 weight matrix-vector multiplications per token.
- Vocab output projection over 248,320 logits.

When profiling `tinygrad` with `DEBUG=2`, we observe that **1,221 individual kernels** are executed in 7 CUDA graph batches per decode step, consuming **5.85 ms** of GPU compute time. In contrast, `llama.cpp` spends ~3.3 ms on matrix-vector multiplications and ~1.4 ms on the recurrent state updates.

The gap breaks down into two distinct bottlenecks:

| Component | `tinygrad` Latency (H100) | `llama.cpp` Latency (H100) | Deficit | Root Cause |
| :--- | :---: | :---: | :---: | :--- |
| **Q8_0 GEMV Projections** | ~3.80 ms | ~3.30 ms | **~0.50 ms** | Scalar 16-bit loads (`_nv_ldcs16`) in `_q8_0_decode_kernel` vs 128-bit vectorization |
| **GatedDeltaNet Recurrent Layers (24x)** | ~2.10 ms | ~1.30 ms | **~0.80 ms** | Decomposed into ~800 elementwise/conv/norm kernels with DRAM roundtrips |
| **Graph Overhead & Argmax** | ~0.24 ms | ~0.15 ms | **~0.09 ms** | Activation quantization & reduction graph dispatch |
| **Total Decode Latency** | **6.14 ms** | **4.75 ms** | **1.39 ms** | Target: Shave 1.4 ms to reach **212+ tok/s** |

---

## 2. Action 1: Vectorized Q8_0 Matrix-Vector GEMV (`nv_linear_q8_0`)

### The Problem in Current Code
Inspect [`tinygrad/llm/kernels/nv.py`](file:///Users/deven367/projects/agent-handoffs/tinygrad/llm/kernels/nv.py#L90-L115):
```python
@functools.cache
def _q8_0_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    xwords = _load_lanes(xq[token, group, 0], 8)
    base = (output*group_count+group)*Q8_U16_WORDS
    dot = UOp.const(0, dtypes.int32)
    for word_idx in range(8):
      # CRITICAL FLAW: 16-bit scalar loads assembled into 32-bit words!
      word = _nv_ldcs16(raw[base+1+word_idx*2]) | (_nv_ldcs16(raw[base+2+word_idx*2]).cast(dtypes.uint32) << 16)
      dot = _nv_dp4a(word, xwords[word_idx], dot)
    return dot.float() * xd[token, group, 0] * _half(_nv_ldcs16(raw[base]))
  return _decode_linear(out, out_features, group_count, group_dot)
```

1. **Memory Load Instruction Inefficiency**:
   - Each Q8_0 block is 34 bytes (2-byte FP16 scale $d$ + 32 bytes of int8 weights = eight 32-bit words).
   - In `_q8_0_decode_kernel`, `raw` is typed as uint16. To read each 32-bit weight word, it emits **two scalar 16-bit `__ldcs` loads**:
     `_nv_ldcs16(...) | (_nv_ldcs16(...) << 16)`.
   - Across the 8 words, that is **16 scalar memory instructions per block** instead of two 128-bit vector memory instructions (`ld.global.nc.v4.u32`)!
2. **Thread Mapping**:
   - `_decode_linear` assigns 1 warp per output neuron. For `intermediate_size = 12288`, a single warp sequentially executes 384 group dot products (3,072 `__dp4a` operations).

### How to Fix It (Estimated Savings: ~0.45 ms/tok)
1. **Vectorize Memory Loads**:
   - View the weight buffer as blocks of 34 bytes or align blocks to load the 32 weight bytes using **two 128-bit vector loads** (`uint4` / `v4.u32`) per block.
   - Just like we did in `_load_byte` -> 128-bit header vectorization for Q4_K (`tinygrad/llm/kernels/nv_q4k.py`), load the 8 weight words in two 128-bit registers (`w0..w3 = raw_u32[base..base+3]`, `w4..w7 = raw_u32[base+4..base+7]`).
   - Load the 2-byte scale $d$ with a single half load (`ld.global.nc.u16`).
2. **Cooperative Multi-Warp Reduction**:
   - For large output projections ($D=12288$), assign 2 or 4 warps per output row and perform a block-level cross-warp reduction with shared memory, maximizing SM occupancy and HBM saturation.

---

## 3. Action 2: Fused Single-Pass GatedDeltaNet Recurrent Kernel

### The Problem in Current Code
Inspect [`tinygrad/llm/model.py`](file:///Users/deven367/projects/agent-handoffs/tinygrad/llm/model.py#L310-L375) in `GatedDeltaNetBlock._attention`:
```python
    # input processing
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))
    ...
    conv_out = functools.reduce(lambda a,b: a+b, ...)
    q, k = (z.normalize(...) for z in (q, k))
```
During decode ($T=1$):
- 24 separate layers execute this sequence sequentially.
- The recurrent state tensor $S$ has shape `[1, 32, 128, 128]` (32 heads, $V=128$, $K=128$), which is **2,097,152 bytes (2.0 MB)** in float32.
- In `tinygrad`, the state decay, delta rule, and output accumulation roundtrip this 2.0 MB buffer through global DRAM twice per layer ($24 \times 4\text{ MB} = 96\text{ MB}$ of memory traffic per token just for state!).
- In addition, separate kernels are launched for:
  - 1D short convolution (`ssm_conv1d`)
  - Two L2 normalization passes on $q$ and $k$
  - Softplus + elementwise decay computation
  - Recurrent scan update
  - RMSNorm + SiLU output gating

### How `llama.cpp` Solves It
In [`ggml-cuda/gated_delta_net.cu`](file:///N/slate/demistry/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu):
- One fused CUDA kernel `gated_delta_net_cuda` handles the entire recurrent cell:
  - Each CUDA thread block processes a head ($H$).
  - Warps map directly across column tiles.
  - The recurrent state column is read, updated in registers:
    $$S_t = S_{t-1} \odot \alpha + (v - S_{t-1} k) \beta \otimes k^T$$
  - The output token projection $y = S_t q$ is computed immediately using warp shuffles.
  - State $S_t$ is written back to DRAM once.
  - Zero intermediate activation buffers are allocated in global memory.

### How to Fix It in `tinygrad` (Estimated Savings: ~0.80 ms/tok)
1. **Extend `gated_delta_prefill` for Decode ($T=1$)**:
   - `tinygrad` already contains a prototype fused scan in [`tinygrad/llm/kernels/amd.py`](file:///Users/deven367/projects/agent-handoffs/tinygrad/llm/kernels/amd.py#L504) (`_gated_delta_prefill_kernel`).
   - Port and specialize this kernel for NVIDIA Hopper / Ada architectures in `tinygrad/llm/kernels/nv.py`:
     - Set block configuration to 1 warp (32 threads) per head/column.
     - Keep the 128-element column vector in thread registers ($4 \times \text{float32}$ per lane).
     - Inline the scalar decay $\alpha = \exp(\text{softplus}(\dots))$ and beta $\beta = \text{sigmoid}(\dots)$ calculation.
2. **Fuse $Q, K$ L2 Norm**:
   - In `_attention`, $q$ and $k$ are normalized with `.normalize(dim=-1, eps=1e-6)`.
   - Fuse the L2 normalization directly into the recurrent kernel (identical to our `fused_qk_norm` pattern from Qwen 27B).

---

## 4. Action 3: Verify Argmax & Logit Projection

1. Vocabulary size for `mimo-qwen` is **248,320** (same as Qwen 2.5 / 3.8).
2. Ensure our custom two-stage `nv_argmax` kernel (from branch `qwen27b-nv-q8-kernel`) is active during `model.generate()`.
3. In `bench_decode.py`:
   - Verify that logit collection does not pull full 248,320-element float arrays back to host memory.
   - `nv_argmax` computes the argmax entirely on GPU in **28 µs**, eliminating host-device synchronization latency.

---

## 5. Verification Checklist for the Next Agent

- [ ] **Baseline Check**:
  Run `make bench-llama-mimo` on `lair-g6` (expect ~73.5 tok/s) and `make bench-tg` (expect ~60.1 tok/s).
- [ ] **Kernel Census**:
  Run decode with `DEBUG=2` before changes:
  `MODEL=/scratch/local/demistry/models/mimo-qwen-q8_0.gguf PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA DEBUG=2 /l/python3/bin/python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 1`
  Note the exact kernel count (baseline: 1,221 kernels).
- [ ] **Implement Vectorized Q8_0**:
  Replace scalar `_nv_ldcs16` in `_q8_0_decode_kernel` with 128-bit vector loads.
  Re-measure: expect decode time to drop by ~0.45 ms on H100 (~1.2 ms on L40S).
- [ ] **Implement Fused GatedDeltaNet Cell**:
  Replace elementwise scan with fused `nv_gated_deltanet` kernel.
  Re-measure: expect kernel count to drop from >1,000 to <100 per step, and decode time to drop by ~0.8 ms on H100 (~1.8 ms on L40S).
- [ ] **Parity Check**:
  Verify greedy token generation matches `llama.cpp` using greedy rollout comparison.
