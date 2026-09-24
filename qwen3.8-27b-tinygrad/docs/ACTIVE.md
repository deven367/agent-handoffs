# ACTIVE — Current state and next steps

> Snapshot: 2026-09-24. Cooperative Warp Q4_K and Q6_K landed & committed (`0fb19cf04`). Logit parity verified (0.9980 cosine sim). L40S decode steady at 31.10 tok/s (vs llama.cpp 37.95 tok/s). Full 2,060-kernel decode census documented.

## Read first

1. `handoff-2026-09-24-l40s-cooperative-warp-and-kernel-census.md` — **latest: Cooperative Q4_K and Q6_K landed and verified on L40S (`lair-g6`), complete decode kernel census (2,060 kernels/step), L40S vs llama.cpp benchmark, next steps for RMSNorm fusion.**
2. `handoff-2026-09-24-decode-gemv-bandwidth-analysis-and-cooperative-warp.md` — GEMV memory load redundancy root cause analysis, cooperative warp blueprints for Q4_K and Q6_K, multi-warp evaluation findings.
3. `handoff-2026-09-24-activation-memoization-and-h100-baseline.md` — activation memoization landed (-240 kernels), H100 decode baseline, llama.cpp parity table, Q8_K_XL 262K verified.

## Ground truth

- Target Compute Node: `lair-g6` (`node-lair`), NVIDIA L40S 48GB GDDR6 (active SLURM job for `demistry`).
  - *Note*: `lair-g1` (H100 NVL) currently rejects SSH via `pam_slurm_adopt` (no active job allocation).
- tinygrad: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`.
- Source HEAD: `0fb19cf04 perf(nv): cooperative warp GEMV for Q4_K and Q6_K` (pushed to fork).
- Launcher: `/u/demistry/agent-handoffs/Makefile`; `~/Makefile` symlinks to it.

## What works

- **Cooperative Warp GEMV for Q4_K and Q6_K**:
  - `nv_q4k.py`: 32 threads cooperatively load 128B `qs`, 0 duplicate weight loads, 100% 128-byte coalescing.
  - `nv_q6k.py`: 32 threads cooperatively load 128B `ql` and 64B `qh` via `_u16_word`, eliminating 2.67× redundancy.
  - Unit sweeps passed: `sweep_q4k.py` ALL OK, `sweep_q6k.py` ALL OK up to 248320x5120 (lm_head).
  - Model logit parity on L40S: `argmax=271` (Match=True), top-5 identical, cosine similarity **0.99804525** (vs baseline 0.9978).
- **L40S Throughput**:
  - `tinygrad`: **31.10 tok/s (32.15 ms/tok)** steady-state decode.
  - `llama.cpp`: **37.95 tok/s (26.35 ms/tok)**.
  - Tinygrad is at **82.0% of llama.cpp** on L40S.
- **Full Decode Census (2,060 kernels/step)**:
  - Quantized GEMV: `nv_linear_q4_k` (432 calls, ~19.0 ms) + `nv_linear_q6_k` (65 calls, ~3.3 ms) = 22.3 ms (75.5% peak memory bandwidth).
  - Non-GEMV: 1,563 calls taking ~9.85 ms (RMSNorm: 322 calls taking ~3.2 ms; activation quantization: 257 calls taking ~2.6 ms; elementwise/residuals: ~880 calls taking ~3.1 ms; scan/attn: 64 calls taking ~0.95 ms).
- Server: OpenAI-compatible API at `/v1/chat/completions` (pinned at `chunk_size=1`).

## Limitations

- **Non-GEMV Overhead on L40S (9.85 ms / 30.6%)**: Split RMSNorm (`r_16_320` + `E_40_32_4`) and activation quantization add ~5.8 ms of non-GEMV latency.
- Prefill pinned at `cs=1` due to chunked prefill divergence on CUDA.
- `lair-g1` access depends on SLURM queue / job allocation.

## Next steps (ranked)

1. **RMSNorm reduction + scaling fusion:**
   - Fuse `r_16_320` (129 calls) + `E_40_32_4` (193 calls) into a single-pass warp-reduction RMSNorm kernel to avoid DRAM round-trips.
   - Expected savings: **~1.8–2.2 ms/tok** on L40S (brings decode from 32.15 ms $\rightarrow$ ~30.0 ms, ~33.3 tok/s).
2. **Vectorize GEMV loads (128-bit `uint4` / `v4.u32`):**
   - Increase memory bus saturation on L40S from 75% to 85%+.
   - Expected savings: **~2.0–2.5 ms/tok** on L40S.
3. **Re-evaluate on H100 NVL (`lair-g1`):**
   - Run `bench_decode.py 512 20` when SLURM job is allocated on `lair-g1` to measure cooperative warp speedup on 3.9 TB/s HBM.

## Important paths

```text
Model:    /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
tinygrad: /u/demistry/tinygrad-src
launcher: /u/demistry/agent-handoffs/Makefile
scripts:  /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts
```
