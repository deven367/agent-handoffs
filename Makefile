# Unified Launcher & Benchmark Makefile for Qwen3.8-27B (Lair & Quartz)
#
# Cluster & environment detection:
#   make info          print detected cluster, GPU, paths, and environment settings
#
# Server controls:
#   make serve         start llama-server (background, log to $(LOGFILE))
#   make status        print health + GPU memory
#   make logs          tail server log
#   make stop          stop server (SIGINT)
#   make serve-tg      start tinygrad server
#   make status-tg     print tinygrad health + GPU memory
#   make logs-tg       tail tinygrad log
#   make stop-tg       stop tinygrad server
#   make serve-mimo    start llama-server with mimo-qwen (background, log to $(LOGFILE))
#   make status-mimo   print health + GPU memory (alias for make status)
#   make logs-mimo     tail server log (alias for make logs)
#   make stop-mimo     stop server (alias for make stop)
#
# Benchmarking & verification:
#   make test-q4k      run cooperative Q4_K unit test sweep
#   make test-q6k      run cooperative Q6_K unit test sweep
#   make test-units    run both Q4_K and Q6_K unit test sweeps
#   make parity        verify full 27B model logit parity (argmax + top-5)
#   make bench-tg      benchmark tinygrad steady-state decode throughput (ctx=512 steps=20)
#   make bench-llama   benchmark llama.cpp decode throughput (ctx=512 steps=20)
#   make bench-llama-mimo benchmark mimo-qwen on llama.cpp (ctx=512 steps=20)
#   make bench         run llama-bench smoke test

# Cluster auto-detection:
#   Quartz: /N/scratch/demistry exists, or hostname contains quartz
#   Lair: /data/user/demistry or /u/demistry exists
IS_QUARTZ := $(shell if [ -d /N/scratch/demistry ] || echo "$$(hostname -f 2>/dev/null)" | grep -qi quartz; then echo 1; else echo 0; fi)

ifeq ($(IS_QUARTZ),1)
CLUSTER        := quartz
MODEL_DIR      ?= /N/scratch/demistry/models
MODEL          ?= $(MODEL_DIR)/Qwen3.8-27B-UD-Q8_K_XL.gguf
MODEL_Q4       ?= $(MODEL_DIR)/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
MMPROJ         ?= $(MODEL_DIR)/qwen3.8-27b-mmproj-BF16.gguf
LLAMA_DIR      ?= /N/slate/demistry/llama.cpp
SERVER         ?= $(LLAMA_DIR)/build/bin/llama-server
LLAMA_BENCH    ?= $(LLAMA_DIR)/build/bin/llama-bench
TG_CWD         ?= $(HOME)/projects/tinygrad-src
CUDA_PATH      ?= /N/soft/rhel8/cuda/12.6/targets/x86_64-linux
TG_ENV         ?= PYTHONPATH=$(TG_CWD) DEV=CUDA CUDA_PATH=$(CUDA_PATH)
SCRIPTS_DIR    ?= $(shell pwd)/qwen3.8-27b-tinygrad/scripts
else
CLUSTER        := lair
MODEL_DIR      ?= /scratch/local/demistry/models
MODEL          ?= $(MODEL_DIR)/Qwen3.8-27B-UD-Q8_K_XL.gguf
MODEL_Q4       ?= /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
MMPROJ         ?= $(MODEL_DIR)/qwen3.8-27b-mmproj-BF16.gguf
LLAMA_DIR      ?= $(HOME)/llama.cpp
SERVER         ?= $(LLAMA_DIR)/build/bin/llama-server
LLAMA_BENCH    ?= $(LLAMA_DIR)/build/bin/llama-bench
TG_CWD         ?= /u/demistry/tinygrad-src
TG_ENV         ?= PYTHONPATH=$(TG_CWD) DEV=CUDA
SCRIPTS_DIR    ?= $(shell pwd)/qwen3.8-27b-tinygrad/scripts
endif

MODEL_MIMO     ?= $(MODEL_DIR)/mimo-qwen-q8_0.gguf

PORT    ?= 9932
ALIAS   ?= Qwen3.8-27B
DEVICE  ?= CUDA0
# native max 256K (262144); YaRN 1M needs rope scaling + more KV mem
CTX     ?= 262144
# slots; 1 = full $(CTX) per request
NP      ?= 1

# Auto-detect GPU and pick a KV cache type that fits comfortably.
GPU_NAME ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
ifeq ($(findstring H100,$(GPU_NAME)),H100)
KV      ?= q8_0
else
KV      ?= q4_0
endif

# YaRN: auto-extend past native 262144 ctx when CTX > YARN_ORIG
YARN_ORIG ?= 262144
ifeq ($(shell test $(CTX) -gt $(YARN_ORIG) && echo y),y)
YARN_ARGS := --rope-scaling yarn --yarn-orig-ctx $(YARN_ORIG) \
             --rope-scale $(shell python3 -c "print('%.4f' % ($(CTX)/$(YARN_ORIG)))") \
             --override-kv qwen35.context_length=int:$(CTX)
else
YARN_ARGS :=
endif
LOGFILE ?= /tmp/llama-server.log

info:
	@echo "=== Cluster & Environment Configuration ==="
	@echo "Cluster:        $(CLUSTER)"
	@echo "Host:           $$(hostname -f 2>/dev/null || hostname)"
	@echo "GPU:            $(GPU_NAME)"
	@echo "Q4 Model:       $(MODEL_Q4)"
	@echo "Q8 Model:       $(MODEL)"
	@echo "Mimo Model:     $(MODEL_MIMO)"
	@echo "llama.cpp dir:  $(LLAMA_DIR)"
	@echo "tinygrad src:   $(TG_CWD)"
	@echo "tinygrad env:   $(TG_ENV)"
	@echo "Scripts dir:    $(SCRIPTS_DIR)"
	@echo "==========================================="

serve:
	@test -f $(SERVER) || { echo "llama-server not found at $(SERVER)"; exit 1; }
	@if curl -sf localhost:$(PORT)/health >/dev/null; then echo "already running on :$(PORT)"; exit 0; fi
	@echo "GPU [$(GPU_NAME)] -> KV=$(KV) CTX=$(CTX) NP=$(NP)"
	nohup $(SERVER) \
		--model $(MODEL) --mmproj $(MMPROJ) --alias $(ALIAS) --tools all \
		--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
		-ngl 999 --device $(DEVICE) \
		--flash-attn on \
		--cache-type-k $(KV) --cache-type-v $(KV) \
		--ctx-size $(CTX) --batch-size 2048 --ubatch-size 2048 --parallel $(NP) \
		$(YARN_ARGS) \
		--spec-type draft-mtp \
		--port $(PORT) --metrics --no-webui \
		> $(LOGFILE) 2>&1 & echo $$! > /tmp/llama-server.pid
	@for i in $$(seq 1 60); do sleep 2; if curl -sf localhost:$(PORT)/health >/dev/null; then echo "server ready on :$(PORT) ($$i x 2s)"; exit 0; fi; done; echo "startup timeout - check $(LOGFILE)"; exit 1

status:
	@curl -s localhost:$(PORT)/health; echo
	@nvidia-smi --query-gpu=memory.used,memory.total --format=csv

logs:
	tail -f $(LOGFILE)

stop:
	@if [ -f /tmp/llama-server.pid ] && kill -0 $$(cat /tmp/llama-server.pid) 2>/dev/null; then \
		kill -INT $$(cat /tmp/llama-server.pid); \
	elif pgrep -x llama-server >/dev/null; then \
		echo "pidfile stale, killing by process name"; pkill -x llama-server; \
	else \
		echo "no server running"; \
	fi; \
	rm -f /tmp/llama-server.pid

bench:
	@test -f $(LLAMA_BENCH) || { echo "llama-bench not found at $(LLAMA_BENCH)"; exit 1; }
	$(LLAMA_BENCH) -m $(MODEL) -p 512 -n 128 \
		--flash-attn on --cache-type-k $(KV) --cache-type-v $(KV) --ubatch-size 2048

TG_KV      ?= q8_0
TG_KV_FLAG ?= $(if $(filter-out f16,$(TG_KV)),--cache-type $(TG_KV),)
TG_MODEL   ?= $(MODEL)
TG_DEV     ?= CUDA
TG_CTX     ?= 262144
TG_PORT    ?= 8888
TG_LOGFILE ?= /tmp/tinygrad-server.log
TG_PIDFILE ?= /tmp/tinygrad-server.pid

serve-tg:
	@test -f $(TG_MODEL) || { echo "tinygrad model not found at $(TG_MODEL)"; exit 1; }
	@if curl -sf localhost:$(TG_PORT)/health >/dev/null; then echo "tinygrad already running on :$(TG_PORT)"; exit 0; fi
	@cd $(TG_CWD); nohup env $(TG_ENV) python3 -m tinygrad.llm --model $(TG_MODEL) --max_context $(TG_CTX) $(TG_KV_FLAG) --serve $(TG_PORT) > $(TG_LOGFILE) 2>&1 & echo $$! > $(TG_PIDFILE)
	@for i in $$(seq 1 90); do sleep 2; if curl -sf localhost:$(TG_PORT)/health >/dev/null; then echo "tinygrad ready on :$(TG_PORT) ($$i x 2s)"; exit 0; fi; done; echo "startup timeout - check $(TG_LOGFILE)"; exit 1

status-tg:
	@curl -s localhost:$(TG_PORT)/health; echo
	@nvidia-smi --query-gpu=memory.used,memory.total --format=csv

logs-tg:
	tail -f $(TG_LOGFILE)

stop-tg:
	@if [ -f $(TG_PIDFILE) ] && kill -0 $$(cat $(TG_PIDFILE)) 2>/dev/null; then \
		kill -TERM $$(cat $(TG_PIDFILE)); \
	else \
		echo "no tinygrad server running"; \
	fi; \
	rm -f $(TG_PIDFILE)

test-q4k:
	@test -d $(TG_CWD) || { echo "tinygrad-src not found at $(TG_CWD)"; exit 1; }
	$(TG_ENV) python3 $(SCRIPTS_DIR)/test_coop_q4k.py

test-q6k:
	@test -d $(TG_CWD) || { echo "tinygrad-src not found at $(TG_CWD)"; exit 1; }
	$(TG_ENV) python3 $(SCRIPTS_DIR)/sweep_q6k.py

test-argmax:
	@test -d $(TG_CWD) || { echo "tinygrad-src not found at $(TG_CWD)"; exit 1; }
	$(TG_ENV) python3 $(SCRIPTS_DIR)/test_argmax.py

test-units: test-q4k test-q6k test-argmax

token-ab:
	@test -d $(TG_CWD) || { echo "tinygrad-src not found at $(TG_CWD)"; exit 1; }
	@test -f $(MODEL_Q4) || { echo "Model not found at $(MODEL_Q4)"; exit 1; }
	$(TG_ENV) MODEL=$(MODEL_Q4) python3 $(SCRIPTS_DIR)/greedy_token_ab.py

parity:
	@test -d $(TG_CWD) || { echo "tinygrad-src not found at $(TG_CWD)"; exit 1; }
	@test -f $(MODEL_Q4) || { echo "Model not found at $(MODEL_Q4)"; exit 1; }
	$(TG_ENV) MODEL=$(MODEL_Q4) python3 $(SCRIPTS_DIR)/compare_logits.py 1

bench-tg:
	@test -d $(TG_CWD) || { echo "tinygrad-src not found at $(TG_CWD)"; exit 1; }
	@test -f $(MODEL_Q4) || { echo "Model not found at $(MODEL_Q4)"; exit 1; }
	$(TG_ENV) MODEL=$(MODEL_Q4) python3 $(SCRIPTS_DIR)/bench_decode.py 512 20

bench-llama:
	@test -f $(LLAMA_BENCH) || { echo "llama-bench not found at $(LLAMA_BENCH)"; exit 1; }
	@test -f $(MODEL_Q4) || { echo "Model not found at $(MODEL_Q4)"; exit 1; }
	$(LLAMA_BENCH) -m $(MODEL_Q4) -n 20 -p 512 -fa 1

MODEL_05B ?= $(MODEL_DIR)/qwen2.5-0.5b-instruct-q4_k_m.gguf

bench-tg-05b:
	@test -d $(TG_CWD) || { echo "tinygrad-src not found at $(TG_CWD)"; exit 1; }
	@test -f $(MODEL_05B) || { echo "0.5B Model not found at $(MODEL_05B)"; exit 1; }
	$(TG_ENV) MODEL=$(MODEL_05B) python3 $(SCRIPTS_DIR)/bench_decode.py 512 20

bench-llama-05b:
	@test -f $(LLAMA_BENCH) || { echo "llama-bench not found at $(LLAMA_BENCH)"; exit 1; }
	@test -f $(MODEL_05B) || { echo "0.5B Model not found at $(MODEL_05B)"; exit 1; }
	$(LLAMA_BENCH) -m $(MODEL_05B) -n 20 -p 512 -fa 1

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

.PHONY: info serve status logs stop bench serve-tg status-tg logs-tg stop-tg test-q4k test-q6k test-argmax test-units token-ab parity bench-tg bench-llama bench-tg-05b bench-llama-05b serve-mimo status-mimo logs-mimo stop-mimo bench-llama-mimo
