# ACTIVE — Current state and next steps

> Snapshot: 2026-09-06. Priority 1 is Q8_K_XL at native 262K on one L40S.

## Read first

1. `sessions/08-2026-09-06-q8-k-xl-l40s-plan.md` — **primary plan:** quantized KV so Q8_K_XL fits at 262K.
2. `sessions/09-2026-09-06-q4-262k-prefill-plan.md` — **secondary plan:** close the 88× tinygrad prefill gap.
3. `progress.md` § "2026-09-06 — Q8_0 KV cache implementation status" — **current blocker and next steps.**

## Ground truth

- Host: `node-lair`, one L40S with `46068 MiB` / `44.99 GiB`.
- tinygrad: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`.
- Source is clean and pushed at `267fcc7e0 feat(llm): add Q8_0 KV cache support`.
- Launcher: `/u/demistry/agent-handoffs/Makefile`; `~/Makefile` symlinks to it.
- GPU was clean at last check.

## Q8_0 KV cache: what works

- `--cache-type f16|q8_0|q4_0` CLI flag, `TransformerConfig.cache_type`, branched attention.
- Q8_0 quantize/dequantize bit-exact vs numpy; roundtrip rel error 0.4%.
- f16-packed single-store cache works on NV; first token matches FP16.
- Makefile safe defaults: `TG_CTX=4096`, `TG_KV=f16`, `TG_DEV` auto-selects CUDA for quantized KV.

## Q8_0 KV cache: blocker

**int8 storage breaks the rangeify scheduler.** `tinygrad/schedule/rangeify.py:419` fails at `type_verify` with `Ops.RANGE dtypes.weakint` when processing int8 tensors in store/read graphs. Happens on both NV and CUDA backends, with both single-store and two-store approaches.

Current workaround (f16 packed cache) uses **2× FP16 memory** — the opposite of the goal.

## Immediate next action

The next agent must do one of:

1. **Fix the rangeify scheduler** to handle int8 store/read graphs (root cause: `type_verify` at `rangeify.py:419`).
2. **Write a fused tiled CUDA attention kernel** that consumes packed Q8_0 cache directly, bypassing the scheduler. This is plan session 08 Phase 3 and is the highest-leverage path.
3. **Patch the scheduler** to handle the specific `Ops.RANGE dtypes.weakint` failure.

Until one of these lands, Q8_K_XL cannot fit at 262K on the L40S through tinygrad.

## Verified completed work

- Q8_K_XL warmup/generation at context 512: ~29.5 GiB tracked (FP16 KV).
- Q4_K_M full context 262144: warmup passed; server used 36,904 MiB.
- Prompt-plus-completion boundary check in `serve.py`.
- Q8_0 KV cache quantize/dequantize verified bit-exact.
- llama.cpp vs tinygrad benchmark: decode 1.35× gap, prefill 88× gap.

## Important paths

```text
Q8:       /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
Q4:       /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
tinygrad: /u/demistry/tinygrad-src
launcher: /u/demistry/agent-handoffs/Makefile
logs:     /tmp/tinygrad-server.log
self-test: /tmp/test_q8_kv.py
```
