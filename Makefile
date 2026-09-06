# llama-server launcher for Qwen3.8-27B (node-lair)
#   make serve   start server (background, log to $(LOGFILE))
#   make status  print health + GPU memory
#   make logs    tail server log
#   make stop    stop server (SIGINT)
#   make bench   quick llama-bench smoke (pp512/tg128)

MODEL   ?= /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
MMPROJ  ?= /scratch/local/demistry/models/qwen3.8-27b-mmproj-BF16.gguf
PORT    ?= 9932
ALIAS   ?= Qwen3.8-27B
DEVICE  ?= CUDA0
CTX     ?= 262144              # native max; YaRN 1M needs rope scaling + more KV mem
NP      ?= 1                   # slots; 1 = full $(CTX) per request. Raise for concurrent clients (ctx splits per slot)

# Auto-detect GPU and pick a KV cache type that fits comfortably.
#   H100 NVL (94GB): q8_0 (higher precision, plenty of room)
#   L40S/unknown:    q4_0 (safe headroom at 262K ctx)
# Override anytime:  make serve KV=f16   or force the branch:  make serve GPU_NAME="NVIDIA H100 NVL"
GPU_NAME ?= $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
ifeq ($(findstring H100,$(GPU_NAME)),H100)
KV      ?= q8_0
else
KV      ?= q4_0
endif

# YaRN: auto-extend past the native 262144 ctx (qwen35) when CTX > YARN_ORIG.
# The context_length metadata override is REQUIRED: this llama.cpp build caps the
# slot ctx at the unscaled n_ctx_train (262144) otherwise.
# Verified at CTX=1048576 (scale 4) on H100 NVL: 300K-token prompts work, pp 911 t/s.
YARN_ORIG ?= 262144
ifeq ($(shell test $(CTX) -gt $(YARN_ORIG) && echo y),y)
YARN_ARGS := --rope-scaling yarn --yarn-orig-ctx $(YARN_ORIG) \
             --rope-scale $(shell python3 -c "print('%.4f' % ($(CTX)/$(YARN_ORIG)))") \
             --override-kv qwen35.context_length=int:$(CTX)
else
YARN_ARGS :=
endif
LOGFILE ?= /tmp/llama-server.log
SERVER   = $(HOME)/llama.cpp/build/bin/llama-server

# Throughput notes:
#  - MTP speculative decode (~1.75x on this model) is enabled below.
#  - Thinking is ON by default (xhigh effort). To cap/trim thinking tokens, add one of:
#      --reasoning-budget 2048                                  # hard cap on thinking tokens
#      --reasoning-effort medium                                # qualitative effort level
#      --chat-template-kwargs '{"reasoning_effort":"medium"}'   # template-native (unsloth recommended)
#  - preserve thinking across turns (uses more ctx/tokens): --reasoning-preserve
#  - H100 NVL (94GB): f16 KV or bigger ctx fine; keep MTP on.

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
	$(HOME)/llama.cpp/build/bin/llama-bench -m $(MODEL) -p 512 -n 128 \
		--flash-attn on --cache-type-k $(KV) --cache-type-v $(KV) --ubatch-size 2048

TG_MODEL   ?= /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
TG_CTX     ?= 262144
TG_PORT    ?= 8888
TG_LOGFILE ?= /tmp/tinygrad-server.log
TG_PIDFILE ?= /tmp/tinygrad-server.pid
TG_CWD     ?= $(HOME)/tinygrad-src

serve-tg:
	@test -f $(TG_MODEL) || { echo "tinygrad model not found at $(TG_MODEL)"; exit 1; }
	@if curl -sf localhost:$(TG_PORT)/health >/dev/null; then echo "tinygrad already running on :$(TG_PORT)"; exit 0; fi
	@echo "Starting tinygrad on :$(TG_PORT) with $(TG_MODEL)"
	@cd $(TG_CWD); nohup python3 -m tinygrad.llm --model $(TG_MODEL) --max_context $(TG_CTX) --serve $(TG_PORT) > $(TG_LOGFILE) 2>&1 & echo $$! > $(TG_PIDFILE)
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

.PHONY: serve status logs stop bench serve-tg status-tg logs-tg stop-tg
