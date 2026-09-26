# ACTIVE — Mimo-Qwen Project State & Next Steps

> Snapshot: 2026-09-26. Evaluated on NVIDIA H100 SXM5 80GB (`g37`) and NVIDIA L40S 48GB (`lair-g6`).
> Architecture: Qwen 3.5 9B (hybrid: 24 GatedDeltaNet recurrent linear attention layers + 8 full attention layers).
> Model: `mimo-qwen-q8_0.gguf` (8.86 GiB, Q8_0 weights, 32 transformer blocks).

---

## Read First

1. [`roadmap-tinygrad-parity.md`](file:///Users/deven367/projects/agent-handoffs/mimo-qwen/docs/roadmap-tinygrad-parity.md) — **START HERE for Optimization: Technical blueprint for closing the 1.39 ms gap to llama.cpp (Vectorized Q8_0 GEMV & Fused GatedDeltaNet).**
2. [`handoff-2026-09-26-initial-benchmarks-and-metadata-fixes.md`](file:///Users/deven367/projects/agent-handoffs/mimo-qwen/docs/handoff-2026-09-26-initial-benchmarks-and-metadata-fixes.md) — Initial benchmark results across H100 and L40S, GGUF metadata repair details, and Makefile controls.

## 1. Fast-Start Commands (Root Makefile)

```bash
# On Quartz (ssh g37) or Lair (ssh lair-g6):
cd ~/projects/agent-handoffs   # (or /u/demistry/agent-handoffs on Lair)

make info                # Check cluster, GPU, model paths, and llama.cpp/tinygrad settings
make bench-llama-mimo    # Benchmark llama.cpp decode throughput (ctx=512 steps=20, flash-attn)
make serve-mimo          # Start production llama-server with mimo-qwen on unified :$(PORT) (default 9932, full 256K ctx)
make status-mimo         # Check server health + GPU VRAM allocation (alias for make status)
make logs-mimo           # Tail server log (alias for make logs)
make stop-mimo           # Gracefully stop server (alias for make stop)
```

Custom benchmark arguments:
```bash
make bench-llama-mimo N=50 P=1024   # Custom 50 decode steps with 1024 context
```

Running in tinygrad directly:
```bash
# On Quartz (H100):
CUDA_PATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux CPATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux/include \
MODEL=/N/scratch/demistry/models/mimo-qwen-q8_0.gguf PYTHONPATH=~/projects/tinygrad-src DEV=CUDA \
python3 ~/projects/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 20

# On Lair (L40S):
MODEL=/scratch/local/demistry/models/mimo-qwen-q8_0.gguf PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA \
/l/python3/bin/python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 20
```

---

## 2. Benchmark Comparison Table (`mimo-qwen-q8_0.gguf`, ctx=512)

| Platform / GPU | Engine | Steady-State Decode (tok/s) | Latency (ms/tok) | Prompt Processing (`pp512`) | VRAM Used |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **H100 SXM5 80GB** (`g37`) | **`llama.cpp`** (`llama-bench`, `-fa 1 -ngl 999`) | **210.29 ± 2.26** | **4.75 ms** | **8,106.35 ± 728.75** | ~9.2 GB |
| **H100 SXM5 80GB** (`g37`) | **`tinygrad`** (`bench_decode.py`, CUDA) | **162.80** | **6.14 ms** | N/A | ~9.6 GB |
| **L40S 48GB** (`lair-g6`) | **`llama.cpp`** (`llama-bench`, `-fa 1 -ngl 999`) | **73.57 ± 0.44** | **13.59 ms** | **8,354.95 ± 536.62** | ~9.2 GB |
| **L40S 48GB** (`lair-g6`) | **`tinygrad`** (`bench_decode.py`, CUDA) | **60.11** | **16.64 ms** | N/A | ~9.6 GB |

### Performance Delta Summary
- **On H100 SXM5**: `llama.cpp` is **1.29x faster** (+47.5 tok/s, -1.39 ms/tok).
- **On L40S**: `llama.cpp` is **1.22x faster** (+13.5 tok/s, -3.05 ms/tok).

---

## 3. Ground Truth & Host Mappings

- **Quartz Node**: `g37.quartz.uits.iu.edu` (`ssh g37`), NVIDIA H100 SXM5 80GB HBM3 (3,350 GB/s bandwidth).
  - Model Path: `/N/scratch/demistry/models/mimo-qwen-q8_0.gguf`
  - F16 Model Path: `/N/scratch/demistry/models/mimo-qwen-f16.gguf`
  - Slurm Job: Partition `h100-debug`, user `demistry` (`--gres=gpu:1 --mem=128G`).
- **Lair Node**: `lair-g6` (`ssh lair-g6`), NVIDIA L40S 48GB GDDR6 (864 GB/s bandwidth).
  - Model Path: `/scratch/local/demistry/models/mimo-qwen-q8_0.gguf`
  - Python: `/l/python3/bin/python3`
  - tinygrad workdir: `/u/demistry/tinygrad-src`
  - llama.cpp workdir: `/u/demistry/llama.cpp`

---

## 4. Key Findings & Architecture Analysis

1. **GatedDeltaNet Recurrent Layer Fusion**:
   - `qwen35` interleaves **24 GatedDeltaNet linear recurrent layers** with **8 standard attention layers**.
   - `llama.cpp` executes a specialized fused CUDA kernel ([`ggml-cuda/gated_delta_net.cu`](file:///N/slate/demistry/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu)) that updates the recurrent state, applies gate decay, and performs the output projection directly in registers/shared memory.
   - `tinygrad` currently breaks these operations down into **1,221 smaller kernels per step** (batched into 7 CUDA graphs). Intermediate state roundtrips through global memory, adding ~1.4 ms/step overhead on H100 and ~3.0 ms/step on L40S.
2. **GGUF Conversion Quirks & Fixes**:
   - Original HuggingFace config had `num_hidden_layers = 32` and `mtp_num_hidden_layers = 1`.
   - `convert_hf_to_gguf.py` wrote `qwen35.block_count = 33` and 33 items for `qwen35.attention.recurrent_layers`.
   - However, only blocks 0..31 exist in tensor data.
   - Three metadata patches were applied to make the GGUF model loadable:
     1. `qwen35.block_count` -> `32`
     2. `qwen35.attention.recurrent_layers` -> sliced to first 32 booleans (maintaining exact 32-byte data alignment via existing padding bytes)
     3. `qwen35.nextn_predict_layers` -> `0` (disabling non-existent MTP draft head lookup)

---

## 5. Next Steps & Optimization Roadmap for tinygrad

1. **Kernel Fusion for GatedDeltaNet Recurrent Cell**:
   - Fuse the delta state update, decay calculation, and gate projection into a dedicated tinygrad kernel/CUDA graph pass.
   - Target: Shave ~1.0 ms/tok off tinygrad decode latency.
2. **DP4A / Tensor-Core Vectorized Q8_0 Dot Product**:
   - Implement hand-tuned vectorized Q8_0 GEMV in `tinygrad/llm/kernels/nv.py` matching `llama.cpp`'s `mul_mat_vec_q8_0`.
   - Target: Accelerate the remaining 427 Q8_0 matrix-vector projections.
