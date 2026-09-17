# ACTIVE — Current state and next steps

> Snapshot: 2026-09-17. Decode 34.2 tok/s (88% of llama.cpp). Fused GatedDeltaNet scan on CUDA.

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
- Fused `gated_delta_prefill` scan kernel on NV/CUDA (was AMD-only).
- Server: OpenAI-compatible API at `/v1/chat/completions` (streaming + non-streaming).
- Q8_K_XL fits at `max_context=262144`: 37.8 GiB tracked (7.2 GiB headroom).
- Q8_0 KV cache, `--cache-type f16|q8_0|q4_0` CLI flag.
- Custom Q4_K/Q6_K/Q8_0 GEMV kernels for decode.

## Limitations

- **Prefill: 34.2 tok/s vs llama.cpp 2595 tok/s (76× gap).** GEMV kernels don't share weights across tokens; generic matmul path too slow. Needs quantized GEMM kernels.
- Requires `DEV=CUDA` backend (NV rangeify scheduler has int8 issues).
- Server uses `chunk_size=1` (multi-token prefill graph OOMs at cs=32).
- Dequantizes full valid prefix per step. Long-context decode slow without tiled attention.

## Next steps

1. **Quantized GEMM kernels for prefill** — batched Q4_K/Q6_K GEMM (biggest gap).
2. **Flash attention for standard blocks** — enable `flash_attention` on NV (model.py:219, currently AMD-only).
3. **Q8_K_XL server smoke test** at 262K context.
4. **Tiled attention kernel** — consume packed Q8_0 cache directly.

## Benchmark (2026-09-17, Q4_K_M, L40S, f16 KV, ctx=512)

| Engine | Decode tok/s | Prefill tok/s | VRAM |
|---|---:|---:|---:|
| llama.cpp | 38.83 | 2595 | 15.65 GiB |
| tinygrad | 34.17 | 34.17 | 17.17 GiB |

## Important paths

```text
Q8:       /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
Q4:       /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
tinygrad: /u/demistry/tinygrad-src
launcher: /u/demistry/agent-handoffs/Makefile
logs:     /tmp/tinygrad-server.log
```
