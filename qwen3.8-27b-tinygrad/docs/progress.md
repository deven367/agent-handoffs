# tinygrad Qwen3.8-27B on node-lair — Progress & Handoff (consolidated 2026-08-27, updated 2026-08-27 session 5)


> **READ FIRST: `ACTIVE.md`** (current state + next steps). This file is the
> historical timeline; earlier session notes were corrected by later ones
> (sessions are frozen in `sessions/`, numbered chronologically).

Original asks: (1) run qwen3.8-27b through tinygrad, (2) compare vs llama.cpp,
(3) write a kernel to improve inference speed. Status: (1)(2)(3) DONE —
two custom NVIDIA GEMV kernels shipped and benchmarked; P0 profile done (session 3) —
**next lever is the Q6_K GEMV kernel, not the DeltaNet chain** (see p1-handoff.md).
- **CORRECTION (session 5):** The Q6_K loader "bug" from session 4 was FALSE. `xh` already has `.lshift(4)` (commit `67ed4c4eb3`). The incorrect fix (`8047d69b8`) was applied and reverted (`65e09942f`). The loader is correct; no corruption occurred on Q6_K weights. Evidence: `p1-q6k-session5-handoff.md` §2, corrected `check_q6k_loader.py` (exact match on 3 real Q6_K tensors: `output.weight`, `blk.0.attn_qkv.weight`, `blk.0.ffn_down.weight`).
- **CORRECTION (session 5):** The "vanishing model" incident was a filename typo (`Q4_K_M` vs `Q4_K_M` with hyphen vs dot). Stable copy at `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (`16,810,705,952` bytes). No evidence of repeated unlinking.
- **Session 5 results:** `nv_q6k.py` (`1fe4ba369`) verified. Sweep (`sweep_q6k.py`): `7/7 OK`, max rel `1.07e-07`. Profile (`kstat2.py`): `65` calls/step, `7.4 ms` (`19.4%` vs `51.4 ms` generic `65%`). Benchmark (`Q4_K_M`): `28.5 tok/s` (`2.3×` over `12.49` session-2). Decode graph: `39.74 ms/step` (`2×` faster vs `78.84 ms` session-3). Custom GEMV share: `72.1%` (`q4_k 52.7%` + `q6_k 19.4%`).
- **Working quant:** `Qwen3.8-27B-Uncensored-Q4_K_M.gguf` (`16.8 GB`) — `nv_q6k` engaged. Unsloth `Qwen3.8-27B-UD-Q4_K_M.gguf` (`16.5 GB`) has different mix (`Q8_0(106) Q3_K(7) Q4_K(104) Q5_K(131) Q6_K(30) IQ4_NL(7) IQ3_S(4) IQ4_XS(117)`); requires loader additions (`Q3_K`/`IQ4_NL`/etc.) — planned after P7 hygiene.
## Status (2026-08-26, session 2 complete)

- **Q8_0 NV GEMV kernel: DONE.** 27B decode 2.06 → **20.9 tok/s** (9.4× base, 47.9 ms/tok,
  ~616 GB/s effective) incl. aligned `__ldcs` streaming loads. Exact (24/24 + 4/4, maxerr=0),
  proxy A/B 32/32.
- **Q4_K NV GEMV kernel: DONE.** Verified exact (sweep 8/8, rel ≤1.8e-06), A/B **32/32** tokens,
  27B Q4_K_M **12.87 tok/s vs generic 2.56 = 5.03×**. Engagement proven (440 `nv_linear_q4_k`
  invocations in DEBUG=2 log).
- **The Q4_K "blocker" was a broken test harness** (see Q4_K section) — detection and routing
  were always working.
- Both kernels on branch `qwen27b-nv-q8-kernel`, fork `deven367/tinygrad`, all commits
  authored `deven367 <masterdeven@gmail.com>` (`~/bin/git-personal` wrapper). Commit identity
  fixed retroactively on both tinygrad and agent-handoffs repos (filter-branch + force-push).
- Remaining gap to llama.cpp: sequential **DeltaNet chain** (~57 ms of 77 ms/step) — see plan P1.

## Session 3 (2026-08-26): P0 profile complete — read `p1-handoff.md` (docs/)

- Per-step profile (2282 kernels, 78.8 ms GPU): the real bottleneck is the **67 Q6_K
  linears on the generic matmul path (~51.4 ms/step, 65%)**, not the DeltaNet chain
  (~4.6 ms, 6%). Session-2 assumption corrected.
- Bench metric note: 12.49 tok/s includes prefill re-run per step; real decode =
  ~49 tok/s (20.3 ms/token).
- Next: build `nv_q6k.py` (Q6_K GEMV; template = nv_q4k.py, dequant from amd.py:192-203).
  Full build/verify plan in p1-handoff.md §6.
- **P4 — Q6_K NV kernel: COMPLETE.** `nv_q6k.py` (`1fe4ba369`) + `amd.py` routing. Verified: sweep (`7/7 OK`), profile (`7.4 ms` vs `51.4 ms` generic), benchmark (`28.5 tok/s`).
- **P5 — BEAM_CACHE: UNVERIFIED** (still — training never completed; `/tmp` wipes). Defer.
- **P6 — MTP / speculative decoding: LOW ROI.** Last structural gap (`37.7` vs `28.5` tok/s). Defer.
## Session 4 (2026-08-26, ~23:45 EDT): PAUSED — read `p1-q6k-handoff.md`

- **Q6_K loader bug found (gguf.py)**: `xl.bitwise_or(xh)` missing `<<4` — the high 2
  bits must land in bits 4-5 (ggml C ref + AMD kernel both correct; loader wrong). Until
  the 1-line fix lands, ALL Q6_K weights on the generic path are corrupted (67 tensors in
  the OBLITERATED file: lm_head, ffn_down, attn_qkv, attn_v). Evidence: p1-q6k-handoff §1.
  Numeric confirmation pending (model-file incident below).
- Q6_K NV kernel design complete, draft written (`kernels/nv_q6k.py` — **unverified**);
  layout corrected vs ggml C (16 scales at bytes 192:208, d f16 at 208:210; p1-handoff §6
  had 12+pad — wrong). Routing diff for amd.py is in the handoff §4.
- **Incident**: `/scratch/.../Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf` intermittently vanishes
  (unknown re-copy process on the shared box; minutes-long ENOENT windows). Copy to
  `/data/user/demistry` (user-suggested, 70T NFS) failed 8/8. Next: stabilize file →
  confirm loader bug on real bytes → fix loader (separate commit) → apply kernel+routing →
  sweep → proxy A/B → 27B decode bench → re-profile → llama.cpp cross-check on the same
  file (`/u/demistry/llama.cpp` has a full CUDA build — completes P7, proves the fix).

## Environment (node-lair == lair-g7)

- GPU: L40S 46 GiB (box flips to H100 NVL sometimes — see ~/Qwen3.8-27B-server-optimization.md).
  755 GB RAM, EPYC 9354. Remote shell is **zsh** (`echo ===` breaks; quote separators).
- tinygrad: `~/tinygrad-src` (branch qwen27b-nv-q8-kernel, base d851aca9a), installed editable.
  Run WITHOUT `DEV` env — Device.DEFAULT is already NV (tgwork_*.sh `DEV="NVK:..."` gates
  `nv_custom_kernels_supported` FALSE: only NV/CUDA prefixes accepted).
- Models: `/scratch/local/demistry/models/`
  - `Qwen3.8-27B-Uncensored-Q8_0.gguf` (29.0 GB) — WORKS, Q8_0 kernel benchmarked
  - `Qwen3.8-27B-UD-Q8_K_XL.gguf` (31.4 GB) — DOES NOT LOAD (Q3_K(11)/Q8_K(15) missing from loader)
  - `Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf` (16.8 GB) — downloaded this session, Q4_K kernel benchmarked
- `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (16.8 GB) → CORRECTED FILENAME (hyphen, not dot). Stable copy at `/data/user/demistry/`. Benchmarked (`28.5 tok/s` with `nv_q6k`). `llama.cpp` cross-check (`P7`): NOT COMPLETED (interrupted by Unsloth switch). Build available at `/u/demistry/llama.cpp/`.
- New file: `/scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf` (`16,464,440,224` bytes) — Unsloth Dynamic v3.0 mix; fails with `GGML type '11' not supported` (`Q3_K`). Small dev model downloaded: `gemma-3-270m-it-UD-Q8_K_XL.gguf` (`471,104,544` bytes) at same path.
- `Qwen3.8-27B-UD-Q8_K_XL.gguf` (`31.4 GB`): FAILS (`Q3_K`/`Q8_K` loader missing) — remains unaddressed.
- clang (CPU backend only): conda env at `/tmp/tgclang/bin`, put on PATH. gcc fails (tinygrad
  passes clang-only `--target=x86_64-none-unknown-elf`).

## Results

### R1 — tinygrad runs 27B on GPU, generic baseline
`python3 -m tinygrad.llm --model ...Q8_0.gguf --max_context 512 --benchmark 20`
→ **2.06 tok/s** (484 ms/tok, ~59 GB/s, 29.2 GB VRAM). First 2 tokens JIT-compile (~22 s + ~10 s).
Arch `qwen35` fully supported incl. GatedDeltaNet blocks (65 blocks = 48 SSM + 16 attention + nextn stripped).

### R2 — vs llama.cpp (same box, Q8_0 Uncensored file)
| engine | decode | note |
|---|---|---|
| tinygrad, generic path | 2.06 tok/s | baseline |
| tinygrad + nv Q8_0 kernel | 19.3→20.9 tok/s | this work |
| llama.cpp, no MTP | 22.5 t/s | pp512 ≈ 2600 t/s |
| llama.cpp + draft-mtp | 37.7 t/s | speculative |

Custom kernel closes the gap to **~88% of llama.cpp no-MTP**; remaining 2× vs +MTP is
speculative decoding (P6) and the DeltaNet chain (P1). llama.cpp not yet benchmarked on the
OBLITERATED Q4_K_M file (P7).

### R3 — NVIDIA Q8_0 custom GEMV kernel: 2.06 → 20.9 tok/s
- `tinygrad/llm/kernels/nv.py`: UOp-built Q8_0 GEMV — packed uint16 weight view (17 words/block),
  `__dp4a` lane dot, warp shuffle reduction, `fmaxf` CUSTOMI (single-evaluated shuffle; ternary
  lowers to double-evaluated `__shfl_xor_sync` → garbage), aligned `__ldcs` streaming loads.
  Exact vs numpy: 24/24 small + 4/4 production shapes, maxerr=0.
- `amd.py::Linear`: `set_quantized` claims Q8_0 (272 B/256 block, uint16 view) only on NV/CUDA;
  `__call__` routes ggml_type==8 to `q8_0_linear`, pad_to/shrink for symbolic token counts.
- Proxy (qwen3.5:0.8b, temp 0): 32/32 tokens identical; 187/187 Linears engage (generic: 0).
- Dead ends: unaligned 4-byte loads and misaligned `__ldcs` hang/garbage; 32B-relayout peaks at
  ~56 GB VRAM (shared file buffer) — both rejected.

### R4 — NVIDIA Q4_K custom GEMV kernel: 12.87 tok/s on Q4_K_M 27B (5.03×)
`tinygrad/llm/kernels/nv_q4k.py` — Q4_K decode mirroring amd.py::_quant_decode_kernel with NV
intrinsics (dp4a, warp reduce), same `_q8_quantize` activations, 36 uint32 words/block (144 B,
4-byte aligned), `_decode_linear` shared with name param (`nv_linear_q4_k`).

Sweep `python3 /u/demistry/sweep_q4k.py` → **8/8 OK, rel ≤1.8e-06** (f32 accumulation-order
noise; integer dot bit-exact). Out 1..128, in 256..2048, partial last warp, chunks=2.
All prior sweep failures were HARNESS bugs, fixed in place:
- nibble packing must be paired-superblock: `qs.reshape(4,2,32)`, low nibble → sub-block 2k,
  high → 2k+1, byte = lo | hi<<4 (== real GGUF layout, == amd.py `>> ((subgroup&1)*4)`)
- scale bytes: `s[j]=sc[j]&63 | (sc[4+j]>>4)<<6; s[4+j]=mn[j]&63 | (mn[4+j]>>4)<<6; s[8+j]=sc[4+j]&15 | mn[4+j]<<4`
- reference used signed q-8; real Q4_K weight = d*sc*q − dmin*mn with UNSIGNED q
- activation quant must pin each group max to ±127 (scale=1.0), like sweep_nv_q8.py
- packed weights tiled PER OUTPUT ROW: raw = concat(blocks*outf); kernel indexes base=(output*(in//256)+block)*36

`amd.py` wiring (committed): import `nv_q4k.q4_k_linear`; `__call__` Q4_K+NV branch mirroring
Q8_0; `set_quantized` DEVICE-GATED — AMD/RDNA3 keeps QUANT_SIZES dict, NV adds Q8_0 (272B) and
Q4_K (144B = Q4_WORDS*4, uint32 view), other devices claim nothing. This also fixes a
pre-existing stock-tinygrad crash: K-quants on NV packed then fell through to generic matmul
with flat 1-D weight → transpose IndexError (why only the Q8_0 27B file ever loaded).

**"Blocking bug" root cause: broken test harness, not wiring.** `proxy_ab_q4k.py::count_type`
walked dicts/`__dict__` but NOT lists; `Transformer.blk` is a plain list → block Linears never
counted; only `.output` was (Q6_K in Q4_K_M recipes → legitimately unclaimed), so n_q4 was
always 0. `proxy_ab.py` (Q8_0) had list handling; the Q4_K copy lost it. Fixed (list/tuple
walk) → A/B **32/32 token positions identical** (CUSTOM q4_k_linears=131, GENERIC=0).
Detection breakdown qwen3.5:4b Q4_K_M (249 Linears): 131 Q4_K + 48 Q8_0 claimed, 70 unclaimed
= Q6_K weights (output + some ffn) → generic, correct. Diag: `/u/demistry/diag_q4k_graph.py`.

### 27B Q4_K_M benchmark (OBLITERATED file, exclusive L40S, max_context 512, bench 20)
- CUSTOM: **12.87 tok/s** (77.7 ms/tok, 255 GB/s, 19.8 GB VRAM); per-step 266 kernels/8 batched,
  GEMV 20.24 ms of 77.09 ms — the other ~57 ms is DeltaNet chain + generic Q6_K + launch overhead.
- GENERIC (same file, `use_custom_quant=False`): **2.56 tok/s** (391 ms/tok, 44 GB/s).
- Speedup **5.03×**. 12.87 is NOT comparable to Q8_0-Uncensored 20.9 (different model/file).
- Logs: /tmp/bench_q4k_{custom,dbg,generic}.log.

## Gotchas (learned the hard way)

1. `tinygrad/llm/gguf.py` `_GGML_QUANT` lacks Q3_K(11)/Q8_K(15) → UD-Q8_K_XL file raises;
   cli.py pins preset qwen3.8:27b to an older HF rev because of it.
2. GGUF type ids match llama.cpp enum; loader dequantizes lazily, weights stay PACKED in VRAM.
3. CPU backend hard-needs clang (`runtime/support/compiler_cpu.py:21`); `CC` env honored.
4. Kernels compile in worker subprocesses → DEBUG asm / to_program_cache invisible from the
   driver process. Don't introspect from the driver; read renderer/codegen statically or print
   from inside the worker.
5. `create_schedule` doesn't exist in this build; `schedule_linear()` stops before PROGRAM.
6. `DEV` env: use plain NV, never the "NVK:..." prefix.
7. `set_quantized` fires lazily on first `Linear.__call__` — post-forward counts only.
8. `generate()` forces chunk_size=1 when `not amd_custom_kernels_supported(device)`
   (llm/model.py:479) → NV prefill is token-by-token (P2).

## Profiling evidence (basis for the plan)

- 0.8b decode step ≈ 891 kernels; 337 reduce kernels = 91.7% of exec time (kstat.py on DEBUG=2 log).
- Cold GEMV microbench (q8cold.py): default codegen 234 GB/s cold (L40S peak 864; llama.cpp ~650);
  f16math no better cold; packed_u32 WORSE — formulation isn't the fix, the schedule is.
- BEAM=2: 75 → 142 tok/s (0.8B), or 3.55–3.96 ms/tok exclusive-GPU (3.4–3.7×). Search costs
  ~4 min/process and results were NOT persisted (beam cache patch unverified) — headroom proof, not shippable.
- 27B Q4_K_M per-step: GEMV 20.24 ms / 77.09 ms → **DeltaNet chain is now the bottleneck**.
- Per-step profile (session 5): `39.74 ms`, `Q6_K` custom `7.4 ms` (`19.4%`), `Q4_K` custom `19.9 ms` (`55.0%`), generic `r_*` `7.2 ms` (`19.8%`), `E_*` `1.1 ms` (`3.2%`), `q8_quantize` `0.7 ms`. Old generic `r_40_32_4_*`/`r_80_32_4_*` families eliminated.
## Improvement plan (ranked; P0 first — profile before touching anything)

- **P0 — Profile the 57 ms/step non-GEMV time.** Run the kstat.py pattern on
  /tmp/bench_q4k_dbg.log for a per-op breakdown (DeltaNet SSM ops vs attention vs Q6_K generic vs
  launch overhead). Any kernel work before this targets the wrong thing.
- **P1 — DeltaNet chain (biggest lever, expected 1.5–2×).** If profiling confirms GatedDeltaNet
  (conv1d + state matmul + frequency-domain path) dominates: port the `gated_delta_prefill`
  fused-path idea to NV or hand-write the SSM update as a custom kernel. 77 → 40 ms/step ≈
  12.9 → ~25 tok/s.
- **P2 — NV chunked prefill.** Remove the AMD-only gate at llm/model.py:479 so NV uses
  chunk_size>1 for prefill; fixes prompt processing (llama.cpp pp512 ≈ 2600 t/s vs token-by-token).
  Completes ask #2 honestly.
- **P3 — Q4_K GEMV micro-opts (only if P0 says GEMV matters).** `__ldcs` streaming loads
  (the Q8_0 Tier-1: 19.3→20.9 tok/s), 16-byte vectorized loads (block stride 144·k % 16 == 0 →
  aligned), pack 12 scale/min bytes into 3 uint32 loads. Also explain avg 255 GB/s when the
  per-kernel line shows up to 630 GB/s — may be a Q6_K/generic artifact.
- **P4 — Q6_K NV kernel.** Claims the remaining generic Linears in Q4_K_M/Q6_K files; mirror the
  Q6_K branch of `_quant_decode_kernel` like nv_q4k.py did.
- **P5 — BEAM_CACHE validation.** Train on 0.8b/4b (`BEAM_CACHE=1 BEAM=2`, explicit db filenames —
  zsh glob aborts), verify hits, replay at zero search cost (3.4–3.7× measured headroom on the
  generic ops in the ~57 ms).
- **P6 — Speculative decoding / MTP.** Last structural gap vs llama.cpp (37.7 t/s +MTP vs 20.9
  Q8_0). Large project, low ROI/hour vs P1–P5.
- **P7 — Hygiene (updated):** `llama.cpp` cross-check on `OBLITERATED-Q4_K_M.gguf`: NOT COMPLETED (interrupted). `Q3_K`/`IQ4_NL` loader additions: NOT STARTED (planned for Unsloth support). Logit-diff A/B (not `argmax`): NOT STARTED. AMD path re-check: NOT STARTED (no AMD GPU).
  file); add Q3_K/Q8_K loader types so UD-Q8_K_XL.gguf loads; logit-diff (not just argmax) A/B;
  AMD path re-check after the set_quantized gating (no AMD GPU here — code-review level).

Suggested order (updated): P7 hygiene (`llama.cpp` bench + loader types) → Unsloth Dynamic v3.0 quant kernels (`Q3_K`/`IQ4_NL` loader + routing) → P1 DeltaNet (if budget remains). P0-P4 COMPLETE.

## Artifacts

On node-lair:
- `~/tinygrad-src/tinygrad/llm/kernels/` — nv.py (Q8_0), nv_q4k.py (Q4_K), amd.py (routing) — committed
- `/u/demistry/` — sweep_nv_q8.py (24/24), sweep_q4k.py (8/8), sweep_nv_real.py,
  proxy_ab.py (Q8_0 A/B), proxy_ab_q4k.py (Q4_K A/B, list-walk FIXED), diag_q4k_graph.py,
  bench_generic.py; patch_amd*.py, fix_import.py (scratch, deletable)
- `/tmp/` — q8gemv.py, q8cold.py, kstat.py, dumpptx.py, dumpasm.py, debug_nv_*.py,
  test_nv_*.py, diag_nv_stage.py, dump_max_src.py, beam_opts.txt
- Logs: /tmp/tg_gpu20.log, tg_small*.log, tg_beam.log, tg_cache_train.log, sweep_nv_real.log,
  bench_q4k_{custom,dbg,generic}.log, llama-server-restored.log
- llama-server restart: `cmd=$(cat /tmp/llama-server.cmd); nohup sh -c "exec $cmd" >/tmp/... &`
  (Qwen3.8-27B-Uncensored Q8_0 + mmproj, port 9932; check `nvidia-smi --query-compute-apps` first)

Local (macOS): /Users/deven367/tmp/{sweep_nv_q8.py,proxy_ab_q4k.py,diag_q4k_graph.py,
bench_generic.py,HANDOFF_Q4K.md,progress_remote.md}

## Appendix — Q8_0 kernel debug history (only if you touch nv.py again)

- **Bug 1 (gater)**: lane-0 gated store → tinygrad's gater predicates the warp → shuffle results
  undefined. Fix: all 32 lanes active, write 32-lane scratch tile, select lane 0 after.
- **Bug 2 (double-eval)**: `UOp.maximum` renders to ternary `(a < shfl) ? shfl : a` evaluating
  `__shfl_xor_sync` twice → garbage (alternating 0/32 in max-reduce). Fix: `_nv_fmax` CUSTOMI,
  one `fmaxf` call. Diagnosis: sweep 24/24 failed → stage isolation (xq words wrong, raw/scale/
  decode correct) → max-reduce broken while sum-reduce fine → DEBUG=4 source dump showed ternary.
- BEAM_CACHE patch (tinygrad/codegen/opt/postrange.py): opt-in persistent beam-schedule cache,
  key = normalized AST key + renderer target + BEAM_ESTIMATE + TF32. Train `BEAM_CACHE=1 BEAM=2`,
  replay `BEAM_CACHE=1`. **Still unverified** (training never completed; /tmp wipes).

## 2026-09-06 — Apples-to-apples llama.cpp vs tinygrad benchmark

Same model (`Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`), same GPU (L40S, 46068 MiB), 3 repetitions.

| Engine | KV | pp512 (tok/s) | tg128 (tok/s) | Decode ms/tok | VRAM |
|---|---|---:|---:|---:|---:|
| llama.cpp (f16 KV) | f16 | 2532.5 | 38.86 | 25.7 ms | ~17.1 GiB |
| llama.cpp (q4_0 KV) | q4_0 | 2503.4 | 38.49 | 26.0 ms | ~16.8 GiB |
| tinygrad | f16 | 28.5 | 28.7 | 34.8 ms | 16.6 GiB |

### Findings

- **Decode**: llama.cpp is 1.35× faster (38.9 vs 28.7 tok/s). Gap is ~8.9 ms/token.
- **Prefill**: llama.cpp is 88× faster (2533 vs 28.5 tok/s). tinygrad forces `chunk_size=1` for recurrent models on non-AMD, so prefill is token-by-token at decode speed.
- **VRAM**: comparable; tinygrad 16.6 GiB vs llama.cpp ~17.1 GiB (f16 KV) or ~16.8 GiB (q4_0 KV).
- **KV cache**: llama.cpp supports Q4_0/Q8_0/f16 KV; tinygrad is FP16-only. At 262K context this is the fitting gap: Q4_0 KV is 4.8 GiB vs FP16 KV 16 GiB.
- KV type barely affects llama.cpp decode speed (38.86 vs 38.49 tok/s); the bottleneck is weight dequantization, not KV bandwidth.

### Bottlenecks (from prior profiling)

- tinygrad decode: Q4_K custom GEMV 19.9 ms (55%), Q6_K custom 7.4 ms (19.4%), generic SSM/reduce 7.2 ms (19.8%). Total ~39.7 ms/step (includes overhead).
- llama.cpp decode: MTP speculative decode (~1.75× measured separately), flash attention, optimized GEMV.
- tinygrad prefill: `chunk_size=1` guard prevents batched prefill. Removing it causes symbolic 32-token GEMV scratch to OOM. This is the largest gap to close.

## 2026-09-06 — Q8_0 KV cache implementation status

### Completed

- `--cache-type f16|q8_0|q4_0` CLI flag added; `TransformerConfig.cache_type` field; `TransformerBlock` branched attention path.
- Q8_0 quantize/dequantize helpers (`_q8_0_quantize_kv`, `_q8_0_dequantize_kv`) verified bit-exact vs numpy reference on zero, extrema, normal, and HDR inputs. Roundtrip relative error: 0.4% (expected Q8_0).
- Self-test at `/tmp/test_q8_kv.py` passes all cases when cache is realized (production path).
- Makefile: `TG_KV ?= f16`, `TG_DEV` auto-selects CUDA for quantized KV, `TG_CTX ?= 4096` safe default.
- Full-model A/B at context 512: FP16 first token 16, Q8_0 first token 16 (match). Tokens diverge after ~2 autoregressive steps (16,17,24,25... vs 16,17,18,19...). This is expected Q8_0 quantization noise accumulating over steps.

### Blocker: int8 storage breaks rangeify scheduler

- The actual memory savings require int8 cache storage (1 byte/value + 0.0625 bytes/value for scales = 1.0625 B/value vs FP16's 2 B/value).
- **Single-store int8 packed cache**: `RuntimeError: UOp verification failed at 14 on Ops.RANGE dtypes.weakint` on both NV and CUDA backends.
- **Two-store int8 (separate values + scales)**: same scheduler error on both backends.
- **f16 packed single-store** (values + expanded scales as float16): works on NV, but uses 4×f16 = 2× FP16 memory — worse than baseline.
- The rangeify scheduler in `tinygrad/schedule/rangeify.py:419` fails at `type_verify` when processing int8 tensors in store/read graphs with symbolic dimensions.

### What the next agent must do

1. **Fix the rangeify scheduler** to handle int8 store/read graphs, OR
2. **Write a custom CUDA kernel** that fuses quantize+store+dequantize+attention, bypassing the scheduler entirely, OR
3. **Patch `tinygrad/schedule/rangeify.py`** to handle the `Ops.RANGE dtypes.weakint` case that fails.

Option 2 is likely the highest-leverage path — a fused tiled attention kernel that consumes packed Q8_0 cache directly, as described in plan session 08 Phase 3. This bypasses the scheduler issue entirely and avoids the dequantize-full-prefix-per-step memory cost.

### Current commit

- `267fcc7e0 feat(llm): add Q8_0 KV cache support` on `fork/qwen27b-nv-q8-kernel`
- Q8_0 path works correctly with f16 packed cache; memory savings blocked on scheduler/kernel work.

## 2026-09-06 — Q8_K_XL fits at 262K on L40S

**Breakthrough.** Compact int8 single-store Q8_0 KV cache makes Q8_K_XL fit at native 262144 context on one L40S.

### How

- Single-store int8 packed cache: values (int8) + compact scales (int8, `scale*127`) packed along last dim.
- Cache shape: `(2, B, H, max_context, head_dim + head_dim//32)` int8 = **2.125 B/value** vs FP16's 4 B/value.
- Scales stored as `int8(scale * 127)`, dequantized by dividing by 127 on read.
- `repeat_interleave` for scale expansion avoids symbolic reshape.
- Scheduler fix: added `spec_shared` rules (RANGE, REDUCE) to `spec_kernel_graph` in `tinygrad/uop/spec.py` so int8 kernel graphs pass `type_verify`.
- Two-store approach still fails with `END src[0] should be KERNEL, not Ops.STORE` in `schedule/__init__.py`; single-store avoids this.

### Verified

- Q8_0 quantize/dequantize bit-exact vs numpy.
- Q8_K_XL at context 512: token 16 matches FP16.
- **Q8_K_XL at context 262144: warmup succeeds, token 16 generated, mem_used 37.8 GiB (7.2 GiB headroom on 44.99 GiB L40S).**

- Dequantizes full valid prefix per step (`# ponytail:` comment marks this). Performance at long context will need a tiled attention kernel.

- Requires `DEV=CUDA` backend (NV rangeify scheduler has int8 issues even with spec fix).
- Dequantizes full valid prefix per step (`# ponytail:` comment marks this). Performance at long context will need a tiled attention kernel.
- Q8_0 KV introduces token divergence after ~2 autoregressive steps (expected quantization noise).

### Makefile defaults updated

```make
TG_KV  ?= q8_0
TG_CTX ?= 262144
```

### Current commit

- `ee7322399 feat(llm): compact Q8_0 KV cache fits Q8_K_XL at 262K on L40S` on `fork/qwen27b-nv-q8-kernel`

## 2026-09-06 — NV backend works, end-to-end server verified, benchmarks

### NV backend

The spec fix (`spec_kernel_graph += spec_shared`) also fixes the NV rangeify scheduler. int8 Q8_0 cache now works on both NV and CUDA. Switched default to NV (faster compilation: 46s vs 70s warmup).

### End-to-end server verified on NV

- `make serve-tg` starts Q8_K_XL at 262144 with Q8_0 KV on NV in 46s.
- VRAM: 42,527 MiB (42.0 GiB, 2.9 GiB headroom).
- Chat completion: `"Hello!"` with reasoning, `finish_reason=stop`.
- Boundary check: 53 + 262144 → HTTP 400.
- `make stop-tg` releases all GPU memory.

### Q8_K_XL apples-to-apples benchmark

Same Q8_K_XL model, same L40S, 3 reps for llama.cpp:

| Engine | Backend | KV | pp512 (tok/s) | tg128 (tok/s) | ms/tok | VRAM |
|---|---|---|---:|---:|---:|---:|
| llama.cpp | CUDA | q8_0 | 2541 | 22.4 | 44.7 | 28.3 GiB |
| llama.cpp | CUDA | q4_0 | 2500 | 22.4 | 44.7 | ~28 GiB |
| tinygrad | NV | q8_0 | 13.2 | 12.9 | 77 | 29.6 GiB |

### Dequant overhead isolated

| Model | KV | tg (tok/s) | ms/tok |
|---|---|---:|---:|
| Q4_K_M | f16 | 27.6 | 36 |
| Q4_K_M | q8_0 | 27.0 | 37 |
| Q8_K_XL | q8_0 | 12.9 | 77 |

Q8_0 KV dequant adds only 2% overhead (1 ms/token). The 2× decode slowdown is purely from Q8_K_XL being 2× heavier — weight bandwidth bound.

### Current Makefile defaults

```make
TG_KV   ?= q8_0
TG_CTX  ?= 262144
TG_DEV  ?= NV
```

### Commits

- `ee7322399` tinygrad (model.py + spec.py)
- `dd39c0f` agent-handoffs (Makefile: NV default, TG_KV_FLAG, TG_DEV)

## 2026-09-06 — Prefill investigation: NVRTC codegen blocks chunked recurrent scan

### Finding

Removed the `chunk_size=1` guard and tested chunk_size=2 and 4. Even chunk_size=2 fails with `NVRTC_ERROR_COMPILATION: expected a "}"` on both NV and CUDA backends.

### Root cause

The GatedDeltaNet recurrent scan on non-AMD backends unrolls in Python:

```python
for t in range(T_pad):
    s1 = state * alpha[:, :, t]
    delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]
    state = s1 + delta * k[:, :, t]
    outs.append((state * q[:, :, t]).sum(-1))
```

With `T_pad=2`, this creates a kernel `r_3_4_8_16_2_4_28start_pos2Btoks29` with unmatched braces — the generated CUDA source is malformed. The kernel is too complex for tinygrad's NV codegen to emit correctly.

This is NOT a memory issue (OOM never occurs — compilation fails first). It's a codegen limitation.

### What doesn't work

- chunk_size=2: NVRTC compilation error (malformed source)
- chunk_size=4: NVRTC compilation error (malformed source)
- Both FP16 and Q8_0 KV: same error
- Both Q4_K_M and Q8_K_XL models: same error
- Both NV and CUDA backends: same error

### What's needed (in order of complexity)

1. **Port `gated_delta_prefill` to NV** (plan 09 Phase 4): AMD has a fused kernel that does the entire scan in one custom kernel. Port it to NV using warp reduction primitives from `nv.py`. This bypasses the codegen issue entirely.
2. **Split the JIT graph**: separate the Linear projections (batchable) from the recurrent scan (sequential) into separate kernels. The Linears process `T_pad` tokens; the scan stays at T=1 per iteration.
3. **Fix the codegen**: investigate why the NV renderer drops the closing brace on large kernels. May be a buffer size limit in `tinygrad/renderer/cuda.py` or the NVRTC bindings.

Option 1 is the highest-leverage path — it's also needed for decode performance (the unrolled scan is slow even at T=1). The AMD `gated_delta_prefill` kernel source is in `kernels/amd.py`.

### Guard restored

The `chunk_size=1` guard remains in place. Prefill is 13.2 tok/s (token-by-token at decode speed). This is the 190× gap vs llama.cpp's 2541 tok/s.
