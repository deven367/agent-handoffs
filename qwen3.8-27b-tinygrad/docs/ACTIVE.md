# ACTIVE — Current state and next steps

> Snapshot: 2026-09-17. Decode 34.2 tok/s (88% of llama.cpp). Prefill 36.0 tok/s (cs=2). Fused GatedDeltaNet scan on CUDA.

## Read first

1. `handoff-2026-09-17-parity-benchmark.md` — **latest: decode 88% of llama.cpp, prefill gap documented.**
2. `handoff-2026-09-17-blk00-internals-bisect.md` — scan logic verified correct, divergence is JIT noise.
3. `progress.md` § "2026-09-06 — Q8_K_XL fits at 262K on L40S" — Q8_K_XL milestone.

## Ground truth

- Host: `node-lair`, one L40S with `46068 MiB` / `44.99 GiB`.
- tinygrad: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`.
- Source HEAD: `dd1cb574d fix(llm): use chunk_size=1 in serve.py` (pushed to fork).
- Launcher: `/u/demistry/agent-handoffs/Makefile`; `~/Makefile` symlinks to it.

## What works

- Decode: **34.2 tok/s** on Q4_K_M (88% of llama.cpp 38.8 tok/s).
- Prefill: **36.0 tok/s** at cs=2 (1.4% of llama.cpp 2595 tok/s — bottleneck is E_2 kernel count, not GEMV).
- Fused `gated_delta_prefill` scan kernel on NV/CUDA (was AMD-only).
- Server: OpenAI-compatible API at `/v1/chat/completions` (streaming + non-streaming).
- Q8_K_XL fits at `max_context=262144`: 37.8 GiB tracked (7.2 GiB headroom).
- Q8_0 KV cache, `--cache-type f16|q8_0|q4_0` CLI flag.
- Custom Q4_K/Q6_K/Q8_0 GEMV kernels for decode.

## Limitations

- **Prefill: 36.0 tok/s vs llama.cpp 2595 tok/s (72× gap).** Bottleneck is 936 E_2 element-wise kernels (20+ ms inter-kernel overhead), NOT the GEMV (0.073 ms/layer constant). GEMV per-token throughput is 10× better than llama.cpp.
- Requires `DEV=CUDA` backend (NV rangeify scheduler has int8 issues).
- Server uses `chunk_size=2` (cs≥4 causes VRAM pressure from graph intermediates).
- Dequantizes full valid prefix per step. Long-context decode slow without tiled attention.

## Next steps

1. **Port `flash_attention` to NV/CUDA** — eliminates ~480 of 936 E_2 kernels. Est. decode: ~50 tok/s.
2. **Reduce E_2 kernel count via fusion** — 456 remaining from norm/FFN/residual ops.
3. **Per-layer graph compilation** — enable cs=8/16/32 without OOM via intermediate reuse.
4. **Q8_K_XL server smoke test** at 262K context.
5. **Tiled attention kernel** — consume packed Q8_0 cache directly.

## Benchmark (2026-09-17, Q4_K_M, L40S, f16 KV, ctx=512)

| Engine | Decode tok/s | Prefill tok/s | VRAM |
| tinygrad | 34.17 | 36.0 | 17.17 GiB |
## Important paths

```text
Q8:       /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
Q4:       /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
tinygrad: /u/demistry/tinygrad-src
launcher: /u/demistry/agent-handoffs/Makefile
logs:     /tmp/tinygrad-server.log
```
