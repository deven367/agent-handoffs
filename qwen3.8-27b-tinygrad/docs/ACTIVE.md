# ACTIVE — Current state and next steps

> Snapshot: 2026-09-06. Q8_K_XL fits at native 262K on one L40S.

## Read first

1. `progress.md` § "2026-09-06 — Q8_K_XL fits at 262K on L40S" — **current state.**
2. `sessions/08-2026-09-06-q8-k-xl-l40s-plan.md` — primary plan (Phases 0-4 complete).
3. `sessions/09-2026-09-06-q4-262k-prefill-plan.md` — secondary plan: prefill optimization.

## Ground truth

- Host: `node-lair`, one L40S with `46068 MiB` / `44.99 GiB`.
- tinygrad: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`.
- Source is clean and pushed at `ee7322399 feat(llm): compact Q8_0 KV cache fits Q8_K_XL at 262K on L40S`.
- Launcher: `/u/demistry/agent-handoffs/Makefile`; `~/Makefile` symlinks to it.
- Makefile defaults: `TG_KV=q8_0`, `TG_CTX=262144`, `TG_DEV=CUDA`.

## What works

- Q8_K_XL at `max_context=262144`: warmup succeeds, token 16 generated, **37.8 GiB** tracked (7.2 GiB headroom).
- Q8_0 quantize/dequantize bit-exact vs numpy.
- Q8_0 KV token 16 matches FP16 at context 512.
- `--cache-type f16|q8_0|q4_0` CLI flag, `TransformerConfig.cache_type`, branched attention.
- Prompt-plus-completion boundary check in `serve.py`.
- Scheduler fix: `spec_kernel_graph` now includes `spec_shared` rules (RANGE, REDUCE).

## Limitations

- Requires `DEV=CUDA` backend (NV rangeify scheduler has int8 issues).
- Dequantizes full valid prefix per step. Long-context decode will be slow without a tiled attention kernel.
- Q8_0 KV introduces token divergence after ~2 autoregressive steps (expected quantization noise).
- `chunk_size=1` guard still in place for recurrent models on non-AMD. Prefill is token-by-token.

## Next steps

1. **Server smoke test**: `make serve-tg` with Q8_K_XL at 262K, verify API request.
2. **Prefill optimization** (session 09): fix GEMV scratch, re-enable chunked prefill.
3. **Tiled attention kernel**: consume packed Q8_0 cache directly, avoid dequantizing full prefix.
4. **Q4_0 KV**: add for more headroom (~34 GiB projected) if needed.

## Important paths

```text
Q8:       /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
Q4:       /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
tinygrad: /u/demistry/tinygrad-src
launcher: /u/demistry/agent-handoffs/Makefile
logs:     /tmp/tinygrad-server.log
self-test: /tmp/test_q8_kv.py
```
