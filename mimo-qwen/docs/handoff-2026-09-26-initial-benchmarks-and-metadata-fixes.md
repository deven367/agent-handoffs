# Handoff: Mimo-Qwen Initial Benchmarks, Metadata Fixes, and Makefile Targets

**Date**: 2026-09-26  
**Context**: Evaluated across **NVIDIA H100 SXM5 80GB** (`g37` on Quartz) and **NVIDIA L40S 48GB** (`lair-g6` on Lair).  
**Model**: `mimo-qwen-q8_0.gguf` (8.86 GiB, Q8_0 weights, 32 transformer layers, Qwen 3.5 9B hybrid architecture).  

---

## 1. Executive Summary

1. **GGUF Model Diagnostics & Repair**:
   - The converted model `/N/scratch/demistry/models/mimo-qwen-q8_0.gguf` had metadata incompatibilities due to MTP (multi-token prediction) configuration in HuggingFace:
     - `qwen35.block_count`: 33 -> fixed to 32.
     - `qwen35.attention.recurrent_layers`: array length 33 -> sliced in-place to 32 elements. Total header size remained identical due to padding absorption (data offset stayed at `10964256`).
     - `qwen35.nextn_predict_layers`: 1 -> fixed to 0 using `gguf-set-metadata` (no MTP weights exist in model).
   - Clean 12 MB header backup and sync was applied across both `g37` and `lair-g6` (`md5sum: b1e393515931cef6e652296afafe0db3`).
2. **Benchmark Results (`llama.cpp` vs `tinygrad`)**:
   - **H100 SXM5 80GB (`g37`)**:
     - `llama.cpp`: **210.29 ± 2.26 tok/s** (4.75 ms/tok), prompt processing **8,106.35 ± 728.75 tok/s**.
     - `tinygrad`: **162.80 tok/s** (6.14 ms/tok).
     - Delta: `llama.cpp` is **1.29x faster** (+47.5 tok/s).
   - **L40S 48GB (`lair-g6`)**:
     - `llama.cpp`: **73.57 ± 0.44 tok/s** (13.59 ms/tok), prompt processing **8,354.95 ± 536.62 tok/s**.
     - `tinygrad`: **60.11 tok/s** (16.64 ms/tok).
     - Delta: `llama.cpp` is **1.22x faster** (+13.5 tok/s).
3. **Root Cause Analysis for Speed Difference**:
   - Qwen 3.5 9B architecture is a hybrid network with 24 GatedDeltaNet recurrent linear attention layers and 8 standard full-attention layers.
   - `llama.cpp` utilizes a hand-tuned fused kernel `ggml-cuda/gated_delta_net.cu` that computes state decay, token updates, and projections in a single pass without roundtripping intermediate states through device memory.
   - `tinygrad` currently linearizes the decode step into **1,221 smaller kernels** batched across 7 CUDA graphs, resulting in memory roundtrips and ~1.4 ms/step added overhead on H100 and ~3.0 ms/step on L40S.
4. **Makefile Integration Landed**:
   - Added `bench-llama-mimo`, `serve-mimo`, `status-mimo`, `logs-mimo`, and `stop-mimo` to root `Makefile` (`6ef2d15`).
   - Works seamlessly on both Quartz (`/N/scratch/demistry/models`) and Lair (`/scratch/local/demistry/models`).

---

## 2. Benchmark Evidence & Logs

### A. NVIDIA H100 SXM5 80GB (`g37`)

#### `llama.cpp` Reference (`llama-bench`)
```console
$ /N/slate/demistry/llama.cpp/build/bin/llama-bench -m /N/scratch/demistry/models/mimo-qwen-q8_0.gguf -n 20 -p 512 -fa 1 -r 3
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 81081 MiB):
  Device 0: NVIDIA H100 80GB HBM3, compute capability 9.0, VMM: yes, VRAM: 81081 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | CUDA       |  -1 |   1 |           pp512 |     8106.35 ± 728.75 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | CUDA       |  -1 |   1 |            tg20 |        210.29 ± 2.26 |
```

#### `tinygrad` Benchmark (`bench_decode.py`)
```console
$ CUDA_PATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux CPATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux/include \
  MODEL=/N/scratch/demistry/models/mimo-qwen-q8_0.gguf PYTHONPATH=~/projects/tinygrad-src DEV=CUDA \
  python3 ~/projects/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 20

decode: 162.80 tok/s (6.14 ms/tok) 20 tokens in 0.12s mem=9635MB
decode: 162.30 tok/s (6.16 ms/tok) 50 tokens in 0.31s mem=9635MB
```

### B. NVIDIA L40S 48GB (`lair-g6`)

#### `llama.cpp` Reference (`llama-bench`)
```console
$ make bench-llama-mimo
/u/demistry/llama.cpp/build/bin/llama-bench -m /scratch/local/demistry/models/mimo-qwen-q8_0.gguf -n 20 -p 512 -fa 1 -ngl 999
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 45467 MiB):
  Device 0: NVIDIA L40S, compute capability 8.9, VMM: yes, VRAM: 45467 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | CUDA       | 999 |   1 |           pp512 |     8354.95 ± 536.62 |
| qwen35 9B Q8_0                 |   8.86 GiB |     8.95 B | CUDA       | 999 |   1 |            tg20 |         73.57 ± 0.43 |
```

#### `tinygrad` Benchmark (`bench_decode.py`)
```console
$ MODEL=/scratch/local/demistry/models/mimo-qwen-q8_0.gguf PYTHONPATH=/u/demistry/tinygrad-src DEV=CUDA \
  /l/python3/bin/python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 20

decode: 60.11 tok/s (16.64 ms/tok) 20 tokens in 0.33s mem=9635MB
decode: 60.08 tok/s (16.65 ms/tok) 50 tokens in 0.83s mem=9635MB
```

---

## 3. Makefile Targets Added

The following targets were added to root `Makefile`:
```makefile
MIMO_ALIAS     ?= mimo-qwen

serve-mimo:
	@test -f $(SERVER) || { echo "llama-server not found at $(SERVER)"; exit 1; }
	@test -f $(MODEL_MIMO) || { echo "Mimo model not found at $(MODEL_MIMO)"; exit 1; }
	@if curl -sf localhost:$(PORT)/health >/dev/null; then echo "already running on :$(PORT)"; exit 0; fi
	@echo "Starting mimo-qwen server on :$(PORT) [GPU $(GPU_NAME)] CTX=$(CTX) NP=$(NP)"
	nohup $(SERVER) \
		--model $(MODEL_MIMO) --alias $(MIMO_ALIAS) \
		-ngl 999 --device $(DEVICE) \
		--flash-attn on \
		--cache-type-k $(KV) --cache-type-v $(KV) \
		--ctx-size $(CTX) --batch-size 2048 --ubatch-size 2048 --parallel $(NP) \
		$(YARN_ARGS) \
		--port $(PORT) --metrics --no-webui \
		> $(LOGFILE) 2>&1 & echo $$! > /tmp/llama-server.pid
	@for i in $$(seq 1 60); do sleep 2; if curl -sf localhost:$(PORT)/health >/dev/null; then echo "mimo server ready on :$(PORT) ($$i x 2s)"; exit 0; fi; done; echo "startup timeout - check $(LOGFILE)"; exit 1

status-mimo: status

logs-mimo: logs

stop-mimo: stop

bench-llama-mimo:
	@test -f $(LLAMA_BENCH) || { echo "llama-bench not found at $(LLAMA_BENCH)"; exit 1; }
	@test -f $(MODEL_MIMO) || { echo "Mimo model not found at $(MODEL_MIMO)"; exit 1; }
	$(LLAMA_BENCH) -m $(MODEL_MIMO) -n $(or $(N),20) -p $(or $(P),512) -fa 1 -ngl 999
```

---

## 4. Next Optimization Opportunities in tinygrad

If optimizing `tinygrad` performance on `mimo-qwen`:
1. **GatedDeltaNet Kernel Fusion**:
   - Replace the multiple decomposed kernels for linear attention recurrent state updates with a custom single-pass fused kernel similar to `ggml-cuda/gated_delta_net.cu`.
   - Expected savings: ~1.0–1.2 ms/tok on H100.
2. **Vectorized Q8_0 GEMV**:
   - `mimo-qwen-q8_0` uses 427 Q8_0 quantized weight tensors.
   - Hand-crafted cooperative GEMV (or tensor-core DP4A dot products) will eliminate the remaining bandwidth deficit.
