# Handoff — Decode GEMV Bandwidth Root Cause Analysis & Cooperative Warp Blueprint (2026-09-24)

Prior handoffs: `handoff-2026-09-24-activation-memoization-and-h100-baseline.md`, `ACTIVE.md`.

---

## 1. Executive Summary & Current Status

* **Hardware / Host**: `lair-g1` (`ssh lair-g1`), NVIDIA H100 NVL 94GB (SM 9.0).
* **Workdir**: `/u/demistry/tinygrad-src` on branch `qwen27b-nv-q8-kernel` (HEAD `058d3fdfd`).
* **Environment**: `PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA` (**do NOT set `CUDA_PATH`**).
* **Decode Throughput Baseline**:
  * `tinygrad`: **41.82 tok/s (23.91 ms/tok)**, VRAM 17,044 MB (`bench_decode.py 512 20`).
  * `llama.cpp`: **71.67 tok/s (13.95 ms/tok)**, VRAM 15.65 GiB (`llama-bench`, flash-attn on, f16 KV).
  * **Gap**: 9.96 ms/tok (tinygrad is ~58.3% of llama.cpp speed).
* **Parity & Correctness Status**:
  * Quantitative logit comparison against `/u/demistry/logits_baseline.npy`:
    * `argmax`: **271** (Match = True).
    * `top-5`: Identical `[(271, 33.6878), (25, 20.9662), (11751, 18.4337), (248044, 17.45), (198, 16.6945)]`.
    * `cosine similarity`: **0.99779664** (exceeds parity threshold $\ge 0.9975$).
  * Unit test sweeps:
    * `python3 /u/demistry/sweep_q4k.py` -> **ALL OK**.
    * `python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/sweep_q6k.py` -> **ALL OK**.
  * Working tree on `lair-g1:/u/demistry/tinygrad-src` is clean.

---

## 2. Profiling Breakdown: Where the 23.91 ms Goes

A single decode step was profiled on H100 NVL using `DEBUG=2` and kernel execution census:

| Kernel Family | Call Count / Step | Total Time (ms) | Time / Call (μs) | % of Decode GPU Time | Notes |
|---|---:|---:|---:|---:|---|
| **`nv_linear_q4_k`** | **424** | **15.45 ms** | 36.45 μs | **35.2%** | Primary bottleneck (GEMV) |
| **`nv_linear_q6_k`** | **61** | **4.46 ms** | 73.13 μs | **10.1%** | 2.0× slower per call than Q4_K |
| **Generic RMSNorm** (`r_16_320` + `E_40_32_4`) | **316** | **3.68 ms** | 11.65 μs | **8.3%** | Split reduce + scale |
| **Generic Elementwise / Res** | ~300 | ~3.50 ms | 11.67 μs | ~8.0% | Activations, residuals, silu |
| **`nv_q8_quantize`** | **252** | **2.14 ms** | 8.49 μs | **5.0%** | Memoized (down from 497) |
| **Attention & Scan** (`flash_decode_partial` + `gated_delta_prefill`) | **64** | **0.95 ms** | 14.84 μs | **2.2%** | Highly efficient |

### Critical Finding
The quantized GEMV kernels (`nv_linear_q4_k` + `nv_linear_q6_k`) consume **19.91 ms out of 23.91 ms (83.3% of total decode kernel time)**!
Closing the gap to `llama.cpp` requires optimizing GEMV memory bandwidth utilization.

---

## 3. Investigation: Multi-Warp Row Tiling

We evaluated multi-warp row tiling (`nwarps = 2` and `nwarps = 4` in `tinygrad/llm/kernels/nv.py::_decode_linear`).

* **Bug resolved**: Tinygrad maps local axes sequentially (`lane = AxisType.LOCAL(32)` must be axis 1 so `threadIdx.x = 32`; `warp = AxisType.LOCAL(nwarps)` must be axis 2 so `threadIdx.y = nwarps`).
* **Correctness**: Passed all unit tests and full logit parity (`argmax 271`, `cosine similarity 0.9978`).
* **Throughput result**:
  * `nwarps = 1`: **23.91 ms/tok (41.82 tok/s)**
  * `nwarps = 2`: **24.36 ms/tok (41.05 tok/s)**
  * `nwarps = 4`: **24.34 ms/tok (41.09 tok/s)**
* **Root cause for no speedup**: Assigning different output rows to different warps in the same thread block does not reduce weight memory traffic. Each warp still independently reads its own weight rows from memory/L2, partitioning the L1 cache and reducing grid wave efficiency on small projections (`attn_v` has `out=1024`, meaning only 256 thread blocks with `nwarps=4`).
* **Verdict**: `nv.py` was kept at `nwarps = 1`.

---

## 4. Root Cause Discovered: Memory Redundancy & Access Patterns

We compared the CUDA kernel implementation in `llama.cpp` (`/u/demistry/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh`) against tinygrad's `nv_q4k.py` and `nv_q6k.py`.

### A. Q4_K Comparison

* **GGML Block Layout**: 256 weights = 144 bytes: `d`(2), `dmin`(2), `scales`(12), `qs`(128 bytes = 32 uint32 words).
* **Tinygrad Current Kernel (`nv_q4k.py`)**:
  * Each thread computes a 32-weight subgroup independently (`group = lane + chunk * 32`).
  * In a Q4_K block, subgroup $2k$ (low nibbles) and subgroup $2k+1$ (high nibbles) share 8 uint32 words of `qs`.
  * Lane 0 (subgroup 0) and Lane 1 (subgroup 1) **both issue 8 uint32 loads for the exact same 8 memory addresses** `raw[base + 4 + 0..7]`.
  * Lanes 2 and 3 duplicate words 8..15; lanes 4 and 5 duplicate words 16..23; lanes 6 and 7 duplicate words 24..31.
  * **Result**: Weight data is loaded with **$2.0\times$ redundancy** (1024 bytes loaded from L1/subsystem for 512 bytes of data per 4 blocks). Furthermore, access across the warp has stride, leading to uncoalesced memory requests.
* **`llama.cpp` Cooperative Warp (`vecdotq.cuh:vec_dot_q4_K_q8_1` / `mul_mat_vec_q`)**:
  * The threads in a warp cooperate across the 256-weight block.
  * In `mul_mat_vec_q`: Thread $t \in [0, 31]$ loads contiguous word `raw[base + 4 + t]`.
  * **Zero redundant loads**: The entire 128 bytes of `qs` for the block is fetched in a **single 128-byte coalesced bus transaction**.
  * Thread $t$ handles 1 word from the even subgroup ($2 \cdot (t // 8)$) and 1 word from the odd subgroup ($2 \cdot (t // 8) + 1$).
  * 100% memory bus efficiency, zero duplicate cache requests.

### B. Q6_K Comparison

* **GGML Block Layout**: 256 weights = 210 bytes: `ql`(128 bytes), `qh`(64 bytes), `scales`(16 bytes), `d`(2 bytes).
* **Tinygrad Current Kernel (`nv_q6k.py`)**:
  * Each group issues 34 scalar 16-bit loads (`_nv_ldcs16`).
  * Threads 0 and 2 load the same `ql` words; threads 0, 1, 2, 3 load the same `qh` words.
  * **$2.67\times$ memory load redundancy** + 34 scalar load instructions per group, resulting in `nv_linear_q6_k` taking 73.1 μs/call (2.0× slower than Q4_K).
* **`llama.cpp` Cooperative Warp (`vecdotq.cuh:vec_dot_q6_K_q8_1`)**:
  * 32 threads cooperate on the 256 weights ($QR6\_K = 2$ dot products per thread).
  * Coalesced 32-bit loads for `ql` and `qh`, eliminating redundancy.

---

## 5. Architectural Blueprint: Cooperative Warp Q4_K

### Mathematical Invariance
Let $B$ be a 256-weight block comprising 8 subgroups $s \in [0, 7]$, each with 8 words $w \in [0, 7]$:
$$\text{Total} = \sum_{s=0}^7 (\text{dot}[s] \cdot d \cdot \text{scale}[s] - \text{qsum}[s] \cdot d_{min} \cdot \text{min}[s]) \cdot xd[s]$$
Because multiplication distributes over addition:
$$\text{Total} = \sum_{\text{lane}=0}^{31} \left( v_{\text{even}}(\text{lane}) + v_{\text{odd}}(\text{lane}) \right)$$
Where for lane $t \in [0, 31]$:
* `pair = t // 8` (in $\{0, 1, 2, 3\}$)
* `w = t % 8` (in $\{0, \dots, 7\}$)
* `s_even = 2 * pair`, `s_odd = 2 * pair + 1`
* `w_raw = raw[base + 4 + t]` (Thread $t$ loads exactly 1 32-bit word, perfectly coalesced across $t \in [0, 31]$)
* `x_even = xq[token, block * 8 + s_even, w]` (1 uint32 activation word)
* `x_odd  = xq[token, block * 8 + s_odd,  w]` (1 uint32 activation word)
* Thread $t$ computes:
  ```python
  w_even = w_raw & 0x0f0f0f0f
  w_odd  = (w_raw >> 4) & 0x0f0f0f0f
  dot_even  = _nv_dp4a(w_even, x_even, 0)
  qsum_even = _nv_dp4a(0x01010101, x_even, 0)
  dot_odd   = _nv_dp4a(w_odd, x_odd, 0)
  qsum_odd  = _nv_dp4a(0x01010101, x_odd, 0)

  v_even = (dot_even.float() * d * sc_even.float() - qsum_even.float() * dmin * m_even.float()) * xd_even
  v_odd  = (dot_odd.float()  * d * sc_odd.float()  - qsum_odd.float()  * dmin * m_odd.float())  * xd_odd
  acc = acc + v_even + v_odd
  ```
* Across the row: Loop `for block in range(num_blocks): acc += ...`
* At the very end of the row: `total = _warp_reduce(acc)`. Only **ONE warp reduction per row**, instead of reductions per chunk!

### Expected Performance Gain
* **Weight memory loaded**: Cut by **50%** (144 bytes loaded vs 288 bytes loaded per block).
* **Bus coalescing**: 100% 128-byte transactions.
* **Latency reduction on `nv_linear_q4_k`**: ~15.45 ms $\rightarrow$ ~9.5–10.0 ms (**~5.5 ms/tok saved**).
* Decode throughput jumps from 41.8 $\rightarrow$ **~54–56 tok/s**.

---

## 6. Next Steps for Next Agent (Ranked Priority)

### Step 1: Implement Cooperative Warp Q4_K (`nv_q4k.py`)
1. Edit `tinygrad/llm/kernels/nv_q4k.py`:
   - Replace the subgroup-per-thread chunked loop with the cooperative block loop.
   - Lane $t$ reads `raw[base + 4 + lane]` and pairs activations `x_even` and `x_odd`.
   - Accumulate `v_even + v_odd` over all blocks in the row.
   - Perform single `_warp_reduce(acc)` at the end and store to `out[token, output, lane]`.
2. Verify correctness:
   ```bash
   ssh lair-g1 "PYTHONPATH=/u/demistry/tinygrad-src python3 /u/demistry/sweep_q4k.py"
   ssh lair-g1 "PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/compare_logits.py 1 --compare /u/demistry/logits_baseline.npy"
   ```
3. Benchmark:
   ```bash
   ssh lair-g1 "PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 20"
   ```

### Step 2: Implement Cooperative Warp Q6_K (`nv_q6k.py`)
1. Edit `tinygrad/llm/kernels/nv_q6k.py`:
   - Replace the 34 scalar `_nv_ldcs16` loads with cooperative warp loads (32 threads per block, $QR6\_K = 2$ dot products per thread, matching `vecdotq.cuh:vec_dot_q6_K_q8_1`).
   - Note on alignment: A 32-bit load at byte offset 210 in odd blocks causes a CUDA misaligned address error. Handle odd blocks with 4-byte aligned base loads and shift extraction or block pairing.
2. Verify correctness:
   ```bash
   ssh lair-g1 "PYTHONPATH=/u/demistry/tinygrad-src python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/sweep_q6k.py"
   ssh lair-g1 "PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/compare_logits.py 1 --compare /u/demistry/logits_baseline.npy"
   ```
3. Expected gain: `nv_linear_q6_k` drops from 4.46 ms $\rightarrow$ ~2.0 ms (**~2.4 ms/tok saved**). Decode reaches **~62–64 tok/s**.

### Step 3: Fuse RMSNorm (`r_16_320` + `E_40_32_4`)
1. Profile shows 316 RMSNorm kernels taking 3.68 ms/step.
2. Fuse the reduction and normalization scale into a single-pass custom kernel or compiler pattern.
3. Expected gain: **~1.8 ms/tok saved**. Decode reaches **~68–71 tok/s** (on par with llama.cpp's 71.67 tok/s).

---

## 7. Key Paths & Commands Reference

```bash
# Model files
Q4_MODEL="/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
LOGITS_BASELINE="/u/demistry/logits_baseline.npy"

# Git repository (tinygrad fork)
cd /u/demistry/tinygrad-src
# Git author identity for commits
~/bin/git-personal commit -m "..."

# Unit sweeps
PYTHONPATH=/u/demistry/tinygrad-src python3 /u/demistry/sweep_q4k.py
PYTHONPATH=/u/demistry/tinygrad-src python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/sweep_q6k.py

# End-to-end logit diff
PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/compare_logits.py 1 --compare /u/demistry/logits_baseline.npy

# Full decode benchmark
PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 20
```
