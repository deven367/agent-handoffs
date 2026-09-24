# Handoff — Activation memoization, H100 parity, and Q8_K_XL 262K verification (2026-09-24)

Prior handoffs: `handoff-2026-09-22-chunked-prefill-verdict.md`, `ACTIVE.md`.

## 1. Summary of work completed today (2026-09-24 on `lair-g1` H100 NVL)

1. **Activation quantization memoization landed and verified (commit `058d3fdfd`):**
   - Profiled decode step showed 497 calls of `nv_q8_quantize` per step.
   - Diagnostic trace revealed 124 tensors were quantized 2 to 4 times each across parallel projections (FFN gate/up, attention/SSM projections).
   - Added activation memoization cache in `tinygrad/llm/kernels/nv.py` and `tinygrad/llm/kernels/amd.py`.
   - **Kernel count reduced from 2,397 to 2,157 (-240 redundant kernels eliminated per step)**.
   - `nv_q8_quantize` calls dropped from 497 to 257 (-48%), shaving 1.85 ms of kernel time per step.
   - **Correctness: 100% bit-exact match** vs baseline (`max abs diff: 0.000000e+00`, `cosine similarity: 1.00000000`).
   - Decode throughput improved from 37.6 → **38.47 tok/s** (26.6 ms → 25.99 ms/tok).

2. **llama.cpp head-to-head parity benchmark on exact same H100 GPU and model:**
   - Model: `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (15.65 GiB).
   - Binary: `~/llama.cpp/build/bin/llama-bench`, f16 KV, ubatch 2048, flash-attn on.
   - **llama.cpp decode**: 71.67 ± 0.10 tok/s (13.95 ms/tok).
   - **llama.cpp prefill (pp512)**: 2175.71 ± 94.20 tok/s (0.46 ms/tok).
   - **tinygrad decode**: 38.47 tok/s (25.99 ms/tok) — 53.7% of llama.cpp.
   - **tinygrad prefill**: ~36.0 tok/s (cs=1) — 1.65% of llama.cpp.

3. **Task 5 (Q8_K_XL 262K smoke test) verified on current HEAD:**
   - Model: `/scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf` (30 GB).
   - Command: `Transformer.from_gguf` + `warmup()` + 262144 context + `cache_type="q8_0"` on `DEV=CUDA`.
   - Result: Model load in 24.87s (31.46 GB), warmup in 39.16s (40.60 GB mem used).
   - Successfully generated token 381 in 0.25s with **53.4 GB VRAM headroom** on H100 NVL (94 GB).

4. **Task 3 (logit-diff A/B tool):**
   - Enhanced `compare_logits.py` with portable path resolution (`/data/user/demistry` and `/N/scratch/demistry`), `--save`, `--compare`, and `--diff`.
   - Outputs top-5 tokens + logits, max absolute difference, mean absolute difference, max relative difference, mean relative difference, and cosine similarity.
   - Verified on cs=1 vs cs=2 (divergence quantitatively measured: max abs diff 23.39, cosine similarity 0.8317).

## 2. Updated Parity Table (H100 NVL 94GB, Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf)

| Engine | Decode (tok/s) | Decode (ms/tok) | Prefill pp512 (tok/s) | VRAM |
|---|---:|---:|---:|---:|
| llama.cpp (f16 KV, flash-attn) | 71.67 | 13.95 ms | 2175.7 | 15.65 GiB |
| tinygrad (baseline 09-22) | 37.60 | 26.60 ms | 36.0 (cs=1) | 17.17 GiB |
| tinygrad (today, q8 memoized) | **38.47** | **25.99 ms** | 36.0 (cs=1) | 17.17 GiB |

## 3. Kernel Census Comparison (JIT=0 DEBUG=2, decode step)

| Metric | 09-22 Baseline | 09-24 Memoized | Delta |
|---|---:|---:|---:|
| Total kernels | 2,397 | 2,157 | **-240 (-10.0%)** |
| `nv_q8_quantize` | 497 | 257 | **-240 (-48.3%)** |
| `nv_linear_q4_k` | 432 | 432 | 0 |
| `nv_linear_q6_k` | 65 | 65 | 0 |
| `gated_delta_prefill` | 48 | 48 | 0 |
| `flash_decode_partial` | 16 | 16 | 0 |
| Generic `r_*` reduce ops | 903 | 903 | 0 |
| Generic `E_*` element-wise | 436 | 436 | 0 |

## 4. Next Priorities (Ranked)

1. **RMSNorm reduction + scaling fusion (Top generic families):**
   - `r_16_320` (129 calls, 1.58 ms) + `E_40_32_4` (129 calls, 1.17 ms) = 258 kernels per step.
   - Each RMSNorm in the model currently emits 2 kernels (one reduce, one scale/cast). Fusing reduction and normalization scaling into a single pass would eliminate 129 kernels and ~1.5 ms.
2. **FFN intermediate reduction (`r_136_32_4_5` - 128 calls):**
   - 128 calls per step (~1.21 ms).
3. **P7 hygiene (mechanical unblock for UD files):**
   - Add Q8_K (15) loader in `tinygrad/llm/gguf.py` (`d: float32`, `qs: int8[256]`, `bsums: int16[16]`, block size 292 bytes).
