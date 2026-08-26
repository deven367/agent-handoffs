# tinygrad Qwen3.8-27B on node-lair — Progress & Handoff (consolidated 2026-08-26)

Original asks: (1) run qwen3.8-27b through tinygrad, (2) compare vs llama.cpp,
(3) write a kernel to improve inference speed. Status: (1)(2)(3) DONE —
two custom NVIDIA GEMV kernels shipped and benchmarked; next lever is the DeltaNet chain.

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
  - two mmproj files (vision; irrelevant to tinygrad)
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
- **P7 — Hygiene.** llama.cpp benchmark on the OBLITERATED Q4_K_M file (completes ask #2 for this
  file); add Q3_K/Q8_K loader types so UD-Q8_K_XL.gguf loads; logit-diff (not just argmax) A/B;
  AMD path re-check after the set_quantized gating (no AMD GPU here — code-review level).

Suggested order: P0 → P1 → P2 → P3, then P4/P5 as budget allows.

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
