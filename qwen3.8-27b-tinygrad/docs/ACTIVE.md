# ACTIVE — Current state and next steps

> Snapshot: 2026-09-24. Decode 38.5 tok/s (H100 NVL). Prefill 36.0 tok/s (cs=1 pinned). Flash attention & GatedDeltaNet scan verified. Activation memoization landed (-240 kernels).

## Read first

1. `handoff-2026-09-24-activation-memoization-and-h100-baseline.md` — **latest: activation memoization landed, H100 decode 38.5 tok/s, llama.cpp parity table, Q8_K_XL 262K verified.**
2. `handoff-2026-09-22-chunked-prefill-verdict.md` — cs>=2 numerically wrong, serve.py pinned at cs=1, FA decode verified on CUDA.
3. `handoff-2026-09-17-parity-benchmark.md` — prefill structural gap analysis.

## Ground truth

- Host: `lair-g1` (`node-lair`), NVIDIA H100 NVL with `95830 MiB` / `93.58 GiB`.
- tinygrad: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`.
- Source HEAD: `058d3fdfd perf(kernels): memoize q8_quantize activations across parallel projections` (pushed to fork).
- Launcher: `/u/demistry/agent-handoffs/Makefile`; `~/Makefile` symlinks to it.

## What works

- Decode: **38.47 tok/s** (25.99 ms/tok) on Q4_K_M (H100 NVL; 53.7% of llama.cpp 71.67 tok/s).
- Prefill: **36.0 tok/s** at cs=1 (1.65% of llama.cpp 2175.7 tok/s; cs>=2 stays off due to divergence).
- Fused `gated_delta_prefill` scan kernel on NV/CUDA (48 calls/step).
- Fused `flash_decode_partial` attention kernel on NV/CUDA (16 calls/step).
- Activation memoization (`_q8_cache`): eliminated 240 duplicate `nv_q8_quantize` kernels per step (bit-exact parity: `max_rel=0.00e+00`).
- Server: OpenAI-compatible API at `/v1/chat/completions` (streaming + non-streaming, pinned at `chunk_size=1`).
- Q8_K_XL fits at `max_context=262144`: 40.6 GiB tracked on H100 (53.4 GiB headroom), verified running token generation.
- Q8_0 KV cache, `--cache-type f16|q8_0|q4_0` CLI flag.
- Custom Q4_K/Q6_K/Q8_0 GEMV kernels for decode.
- Quantitative logit diffing tool in `scripts/compare_logits.py` with top-5 + max abs/rel diff and cosine similarity.

## Limitations

- **Prefill: 36.0 tok/s vs llama.cpp 2175.7 tok/s (60× gap).** Structural: no tensor-core matmul for prompt evaluation (GEMV constant 0.073 ms/layer) + per-node latency floor.
- `cs>=2` produces divergent logits (argmax 220 vs correct 271); `serve.py` pinned at `chunk_size=1`.
- Requires `DEV=CUDA` backend (NVRTC works without setting `CUDA_PATH`).

## Next steps (ranked)

1. **RMSNorm reduction + scaling fusion:** `r_16_320` (129 calls) + `E_40_32_4` (129 calls) = 258 kernels/step. Fusing them removes 129 kernels and ~1.5 ms/tok.
2. **FFN intermediate reduction fusion:** `r_136_32_4_5` (128 calls/step, ~1.2 ms).
3. **P7 hygiene:** Q8_K (15) loader in `tinygrad/llm/gguf.py` (`d: float32`, `qs: int8[256]`, `bsums: int16[16]`, 292 bytes/block).
4. **Per-layer graph compilation:** enable memory reuse across layer boundaries.

## Head-to-Head Parity Benchmark (2026-09-24, Q4_K_M, H100 NVL, f16 KV, ctx=512)

| Engine | Decode tok/s | Decode ms/tok | Prefill pp512 tok/s | VRAM |
|---|---:|---:|---:|---:|
| llama.cpp (flash-attn) | 71.67 | 13.95 ms | 2175.7 | 15.65 GiB |
| tinygrad (09-22 HEAD) | 37.60 | 26.60 ms | 36.0 | 17.17 GiB |
| tinygrad (09-24 memoized) | **38.47** | **25.99 ms** | 36.0 | 17.17 GiB |

## Important paths

```text
Q8:       /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
Q4:       /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
tinygrad: /u/demistry/tinygrad-src
launcher: /u/demistry/agent-handoffs/Makefile
logs:     /tmp/tinygrad-server.log
```
