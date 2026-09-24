# ACTIVE — Current state and next steps

> Snapshot: 2026-09-24. Decode 41.8 tok/s (H100 NVL). Prefill 36.0 tok/s (cs=1 pinned). Flash attention & GatedDeltaNet scan verified. GEMV memory redundancy root cause identified; cooperative warp blueprint drafted.

## Read first

1. `handoff-2026-09-24-decode-gemv-bandwidth-analysis-and-cooperative-warp.md` — **latest: GEMV memory load redundancy root cause analysis, cooperative warp blueprints for Q4_K and Q6_K, multi-warp evaluation findings, step-by-step implementation plan.**
2. `handoff-2026-09-24-activation-memoization-and-h100-baseline.md` — activation memoization landed (-240 kernels), H100 decode baseline, llama.cpp parity table, Q8_K_XL 262K verified.
3. `handoff-2026-09-22-chunked-prefill-verdict.md` — cs>=2 numerically wrong, serve.py pinned at cs=1, FA decode verified on CUDA.

## Ground truth

- Host: `lair-g1` (`node-lair`), NVIDIA H100 NVL with `95830 MiB` / `93.58 GiB`.
- tinygrad: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`.
- Source HEAD: `058d3fdfd perf(kernels): memoize q8_quantize activations across parallel projections` (pushed to fork).
- Launcher: `/u/demistry/agent-handoffs/Makefile`; `~/Makefile` symlinks to it.

## What works

- Decode: **41.82 tok/s** (23.91 ms/tok) on Q4_K_M (H100 NVL; 58.3% of llama.cpp 71.67 tok/s).
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

1. **Cooperative Warp Q4_K Kernel (`tinygrad/llm/kernels/nv_q4k.py`):**
   - Eliminate 50% duplicate weight loads and achieve 100% 128-byte coalescing by having all 32 lanes cooperate on each 256-weight block (thread $t$ loads `raw[base + 4 + t]`).
   - Expected savings: **~5.5 ms/tok** (bringing decode down from 23.9 ms $\rightarrow$ ~18.4 ms/tok, ~54 tok/s).
2. **Cooperative Warp Q6_K Kernel (`tinygrad/llm/kernels/nv_q6k.py`):**
   - Replace 34 scalar 16-bit loads with cooperative warp loads.
   - Expected savings: **~2.4 ms/tok** (bringing decode down to ~16.0 ms/tok, ~62–64 tok/s).
3. **RMSNorm reduction + scaling fusion:**
   - Fuse `r_16_320` (127 calls) + `E_40_32_4` (189 calls) into single-pass RMSNorm.
   - Expected savings: **~1.8 ms/tok** (bringing decode down to ~14.2 ms/tok, ~70 tok/s).
4. **P7 hygiene:** Q8_K (15) loader in `tinygrad/llm/gguf.py` (`d: float32`, `qs: int8[256]`, `bsums: int16[16]`, 292 bytes/block).

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
