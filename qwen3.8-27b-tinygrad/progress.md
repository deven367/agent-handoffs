# tinygrad Qwen3.8-27B on node-lair — Progress & Handoff

Date: 2026-08-25. Original asks: (1) run qwen3.8-27b through tinygrad, (2) compare vs llama.cpp,
(3) write a kernel to improve inference speed. Status: (1)(2) DONE, (3) IN PROGRESS — evidence
gathered, implementation not landed. This doc hands off cleanly.

## Environment (node-lair == lair-g7)

- GPU: L40S 46 GiB (this box flips to H100 NVL sometimes — see ~/Qwen3.8-27B-server-optimization.md).
  755 GB RAM, EPYC 9354. Remote shell is **zsh** (`echo ===` breaks; quote separators).
- tinygrad: `~/tinygrad-src`, git `d851aca9a`, installed editable (imports resolve there).
- Models: `/scratch/local/demistry/models/`
  - `Qwen3.8-27B-Uncensored-Q8_0.gguf` (29.0 GB) — **WORKS in tinygrad**
  - `Qwen3.8-27B-UD-Q8_K_XL.gguf` (31.4 GB) — **DOES NOT LOAD** (see gotcha 1)
  - two mmproj files (vision; irrelevant to tinygrad)
- clang (needed only for CPU backend): installed via
  `/l/anaconda3/bin/conda create -y -p /tmp/tgclang -c conda-forge clang_linux-64`
  → `/tmp/tgclang/bin`. Put on PATH for runs. gcc does NOT work (tinygrad passes
  clang-only `--target=x86_64-none-unknown-elf`; `CC=gcc` env is honored but fails).

### Gotchas learned the hard way
1. `tinygrad/llm/gguf.py`: `_GGML_QUANT/_GGML_NATIVE` lack **Q3_K(11)/Q8_K(15)** → UD file raises.
   `llm/cli.py` itself pins preset `qwen3.8:27b` to an older HF rev "because the UD replacement
   uses Q3_K tensors the loader doesn't support". Only the plain Q8_0 file loads.
2. GGUF type ids match llama.cpp's current enum (F32=0, Q8_0=8, BF16=30...). Loader dequantizes
   lazily; weights stay PACKED in VRAM (~29 GB for the 27B — fits L40S).
3. CPU backend hard-needs clang; `CC` env honored (`runtime/support/compiler_cpu.py:21`).
4. Kernels compile in **worker subprocesses** → DEBUG asm output and `to_program_cache`
   are not visible from the driver process. Don't chase this (I wasted turns).
5. `create_schedule` doesn't exist in this build; `Tensor.schedule_linear()` returns graph
   before PROGRAM creation. Use runtime-level tools instead.

## Result 1 — tinygrad runs the 27B on GPU (verified)

```
cd ~/tinygrad-src && PATH=/tmp/tgclang/bin:$PATH python3 -m tinygrad.llm \
  --model /scratch/local/demistry/models/Qwen3.8-27B-Uncensored-Q8_0.gguf \
  --max_context 512 --benchmark 20
```
Steady state (baseline, generic codegen path): **2.06 tok/s** (484 ms/token, ~59 GB/s
effective weight traffic, ~29.2 GB VRAM). Superseded by the custom kernel — see Result 3.
First 2 tokens are JIT compile (~22 s + ~10 s). Arch `qwen35` fully supported incl.
GatedDeltaNet blocks (65 blocks = 48 SSM + 16 attention + nextn stripped).

## Result 2 — comparison vs llama.cpp (same box, same model, from session notes)

| engine | decode | note |
|---|---|---|
| tinygrad, generic path | 2.06 tok/s | fallback codegen path (baseline) |
| tinygrad + nv Q8_0 kernel | **19.3 tok/s** | this work — see Result 3 |
| llama.cpp, no MTP | 22.5 t/s | pp512 ≈ 2600 t/s |
| llama.cpp + draft-mtp | 37.7 t/s | speculative decoding |

Generic-path tinygrad is **~11–18× slower**. No MTP/speculative support — with the custom
kernel the gap shrinks to **1.2× vs llama.cpp no-MTP** (19.3 vs 22.5 t/s) and **2× vs llama.cpp
+ MTP** (37.7 t/s).

## Result 3 — NVIDIA Q8_0 custom GEMV kernel: DONE (2.06 → 19.3 tok/s, 9.4×)

Shipped on branch `qwen27b-nv-q8-kernel` in fork `deven367/tinygrad` (3 commits:
BEAM_CACHE patch, nv.py kernel, amd.py integration), on top of d851aca9a.

- `tinygrad/llm/kernels/nv.py` — UOp-built Q8_0 GEMV: packed uint16 weight view
  (17 words/block), `__dp4a` lane dot, warp shuffle reduction via `__shfl_xor_sync`,
  `fmaxf` CUSTOMI (avoids double-evaluated shuffle from ternary lowering). Exact vs numpy:
  24/24 small + 4/4 production shapes (10240×5120, 5120×17408, 17408×5120×2tok,
  151936×1024), maxerr=0.
- `tinygrad/llm/kernels/amd.py::Linear` — `set_quantized` recognizes Q8_0
  (272 B/256-weight block) only on NV/CUDA (uint16 view, byte-aligned offset);
  `__call__` routes `ggml_type==8` on NV to `q8_0_linear`, with pad_to/shrink for symbolic
  token counts. AMD + generic paths unchanged.
- Proxy equivalence (qwen3.5:0.8b, greedy, temp 0): **32/32 token positions identical**;
  187/187 Linears engage the custom kernel (generic run: 0).
- 27B benchmark (L40S, exclusive, Q8_0 file, max_context 512, bench 20):
  **steady 19.3 tok/s = 51.8 ms/token, ~616 GB/s effective weight traffic**
  (baseline 2.06 tok/s / 59 GB/s → **9.4×**). `nv_linear_q8_0` + `nv_q8_quantize` confirmed
  in the compile log. GEMV bandwidth 616 GB/s ≈ llama.cpp-class (~650); the remaining gap to
  llama.cpp + MTP is the sequential DeltaNet chain (old 484 ms/token: ~420 ms non-GEMV).
- BEAM_CACHE patch (postrange.py) is committed but **unverified** — training never completed
  (slurm GPU-cgroup churn, /tmp wipes). Not needed for the kernel result.
- **Tier 1 follow-up (same branch)**: aligned `__ldcs` streaming loads on the 16 weight
  words + scale (weights read once/token; L2 kept for reused xq/xd). 27B decode
  **51.8 → 47.9 ms/token (19.3 → 20.9 tok/s, +7.6%)**; 0.8B 4.68 → 4.09 ms (+12.6%).
  Still exact (24/24 + 4/4, maxerr=0); proxy A/B 32/32 tokens. Now at **88% of llama.cpp
  decode rate (20.9 vs 23.8 t/s)**. Dead ends explored: unaligned 4-byte loads and
  misaligned `__ldcs` hang or return garbage on this stack; a relayout to aligned 32B
  blocks peaks at ~56 GB VRAM (file buffer is shared across tensors) — both rejected.

## Profiling evidence (what the kernel work must fix)

Proxy model `qwen3.5:0.8b` (same arch family, downloads automatically, <1 GB VRAM — runs
beside llama-server):

- One decode step ≈ **891 kernels**; reduce kernels: 337 count = **91.7 % of exec time**
  (parser: `/tmp/kstat.py /tmp/tg_small_dbg.log`, log produced by `DEBUG=2 ... --benchmark 1`).
- Cold-cache GEMV microbench (`/tmp/q8cold.py`, numerically verified vs numpy, rel_err ≤3e-7;
  variants in `/tmp/q8gemv.py`):
  - default codegen GEMV: **234 GB/s** cold (L40S peak 864; llama.cpp ≈ 650)
  - `f16math` (skip f32 round-trip on scales): no better cold; 2× warm only
  - `packed_u32` vectorized lane-decode: WORSE cold (138 GB/s)
  - ⇒ formulation tweaks don't fix it; the emitted schedule is the problem.
- **BEAM=2 doubles end-to-end**: 75 → 142 tok/s on 0.8B (13.25 → 7.0 ms/token).
  But search costs ~4 min per process start and **results are NOT persisted** (no beam cache
  in this build — grep confirmed). So beam proves headroom; it is not a shippable answer.
- 27B arithmetic: GEMV floor at ~450 GB/s ≈ 64 ms/token; measured 484 ms ⇒ ~420 ms in
  non-GEMV work (DeltaNet sequential chain + launch overhead of ~900 kernels/step scaled up).
  Fixing GEMV alone caps at roughly 5–8× improvement; DeltaNet path is the other lever.
- `generate()` forces `chunk_size=1` when `not amd_custom_kernels_supported(device)`
  (llm/model.py:479) → prefill is token-by-token on NV. Prefill speed is terrible because of this.

## Where the previous agent circled (avoid re-doing this)

Tried to introspect generated PTX to hand-write an NV kernel: `create_schedule` gone →
`schedule_linear()` has no PROGRAM ops pre-realize → `to_program_cache` empty in parent
(worker-process isolation) → DEBUG=4 asm capture failed for same reason. Conclusion: don't
introspect from the driver process; either read `tinygrad/renderer/ptx.py` +
`tinygrad/codegen/*` statically, or print from inside the worker.

## Recommended next steps (ranked)

0. **GPU access**: llama-server (user's, PID changes; find via `nvidia-smi
   --query-compute-apps`) holds ~45 GB and makes benchmarks noisy. Ask user to stop it or
   get a window; clean stop/restart exists (`kill <pid>`; their Makefile has serve/stop but
   this instance wasn't started via make — it was started 16:03 Aug 25 with alias
   Qwen3.8-27B-Uncensored, port 9932).
1. **Harvest beam winners cheaply**: rerun small-model benchmark with `BEAM=2 DEBUG=3`
   (codegen prints `opts: [...]` per kernel at DEBUG≥3, see codegen/__init__.py:419).
   Collect winning applied_opts for the top reduce kernels, then hardcode them as defaults
   (patch KernelInfo defaults or intercept before `to_program`) gated on env e.g. `NVQ8=1`.
   Expected: deterministic ~2× end-to-end, zero search cost. Verify on 27B.
2. **Write real NV quant kernels** mirroring `tinygrad/llm/kernels/amd.py` (house pattern:
   UOp-built programs, `Ops.CUSTOM` for intrinsics, registered like `Linear` in
   llm/model.py:5). First target: Q8_0 GEMV with v4 vector loads + split-K; check whether
   `renderer/ptx.py` passes Ops.CUSTOM text through to PTX. Target ≥450 GB/s cold on
   [10240,5120] using the cold harness `/tmp/q8cold.py`.
3. **DeltaNet/fused-path work**: port `gated_delta_prefill` idea to NV; revisit chunk_size>1
   for prefill. Biggest remaining lever after GEMV.
4. Re-run final numbers: 27B benchmark exclusive-GPU; update the table above.

## Artifacts

On node-lair:
- `/tmp/q8gemv.py` — warm GEMV variant bench (correctness vs numpy included)
- `/tmp/q8cold.py` — cold-cache rotation bench (L2-thrash protocol)
- `/tmp/kstat.py` — aggregates last-step kernel stats from DEBUG=2 logs
- `/tmp/dumpptx.py`, `/tmp/dumpasm.py` — introspection attempts (kept to avoid redoing)
- Logs: `/tmp/tg_gpu20.log` (27B baseline), `/tmp/tg_small*.log`, `/tmp/tg_beam.log`

Local (macOS): `/tmp/q8gemv.py`, `/tmp/q8cold.py`, `/tmp/kstat.py` copies.

## Update — 2026-08-25, second implementation pass

### Current remote working tree

All edits are in `~/tinygrad-src` on node-lair and local mirrors under
`/Users/deven367/tmp/tinygrad-work/`:

- **Modified:** `tinygrad/codegen/opt/postrange.py`
  - Adds an opt-in persistent beam schedule cache.
  - Train: `BEAM_CACHE=1 BEAM=2 ...`
  - Replay without search: `BEAM_CACHE=1 ...`
  - Cache key normalizes `KernelInfo.beam/applied_opts`, and includes AST key, renderer target,
    `BEAM_ESTIMATE`, and TF32 state.
  - **Not verified yet.** The first training command never ran because zsh expanded
    `/tmp/tg-beam-cache.db*` and aborted with `no matches found`. Retry using explicit filenames,
    not a glob:
    `rm -f /tmp/tg-beam-cache.db /tmp/tg-beam-cache.db-shm /tmp/tg-beam-cache.db-wal`.
- **New, unintegrated:** `tinygrad/llm/kernels/nv.py`
  - NVIDIA Q8_0 custom-kernel attempt using `Tensor.custom_kernel`.
  - Activation quantization to 32-value Q8 groups.
  - Packed Q8_0 weight decoding and CUDA `__dp4a`.
  - Warp butterfly reduction via `__shfl_xor_sync`.
  - This is **not wired into `model.py`/`Linear` yet**.
- `tinygrad/renderer/ptx.py` was briefly modified, then fully reverted. The live `NV` backend
  uses tinygrad's CUDA C renderer; PTX renderer changes are unnecessary.

Run `git -C ~/tinygrad-src status --short` before continuing.

### New exclusive-GPU measurements

The resident llama-server was paused after saving its exact argv to `/tmp/llama-server.cmd`.
With exclusive GPU access:

```
BEAM=2 DEBUG=3 python3 -m tinygrad.llm \
  --model qwen3.5:0.8b --max_context 512 --benchmark 4
```

- Steady decode: **3.55–3.96 ms/token = 252–281 tok/s**.
- Previous default: ~13.25 ms/token = 75 tok/s.
- This is a **3.4–3.7× measured improvement** from beam-selected schedules.
- 65 unique winning `opts:` recipes were emitted.
- Log: `/tmp/tg_beam_debug.log`
- Extracted opts: `/tmp/beam_opts.txt`

This is stronger evidence than the earlier contended 142 tok/s result. Persisting/replaying the
beam choices is the fastest credible optimization path.

### NVIDIA Q8_0 kernel verification state

Successful:

- CUDA C `Ops.CUSTOMI` `__dp4a` smoke test passed: `/tmp/test_nv_dp4a.py` → `[30, -4]`.
- Activation Q8 quantizer matches expected packed words and scale exactly:
  `/tmp/debug_nv_q8.py`.
- A CUDA warp-reduction bug was found: a lane-0 gated store caused tinygrad's gater to predicate
  the warp, making shuffle results undefined. Fixed by keeping all 32 lanes active, writing a
  32-lane scratch tile, then selecting lane 0.
- After that fix, deterministic dimension cases pass exactly:
  `/tmp/debug_nv_cases.py`:
  - `(out=1, in=64)`
  - `(out=2, in=32)`
  - `(out=2, in=64)`

Still failing:

- `/tmp/test_nv_q8_0.py` fails exact comparison for random larger cases, starting at
  `(out=64, in=64)`.
- `/tmp/debug_nv_random.py` confirms the random-data mismatch even at `(out=8, in=64)`:
  expected `[-1526, -994, 1798, 707, 1596, 50, -554, 1031]`, actual
  `[-351, 2965, -2930, 4757, 1658, -446, -1750, -237]`. Input Q8 quantization is already
  proven correct, so inspect random packed-weight addressing/loading next.
- Do **not** integrate `nv.py` until random correctness passes.

Likely integration after correctness:

1. Extend `tinygrad/llm/kernels/amd.py::Linear.set_quantized` to recognize Q8_0:
   34 packed bytes per 32 decoded weights; use a `uint16` buffer view (17 words/block).
2. In `Linear.__call__`, route `ggml_type == 8` on NV/CUDA to
   `tinygrad.llm.kernels.nv.q8_0_linear`; retain AMD and generic paths unchanged.
3. Compare greedy logits/tokens against the generic path on `qwen3.5:0.8b`.
4. Benchmark full Qwen3.8-27B only after proxy equivalence.

### Service restoration

The server was restored after the benchmark window.

- Health: `curl -sf localhost:9932/health` → `{"status":"ok"}`
- Restored PID: `2932371`
- VRAM: `43894 MiB`
- Log: `/tmp/llama-server-restored.log`
- Saved argv remains at `/tmp/llama-server.cmd`

For a future restart under zsh:

```
cd ~/llama.cpp
cmd=$(cat /tmp/llama-server.cmd)
nohup sh -c "exec $cmd" >/tmp/llama-server-restored.log 2>&1 </dev/null &
```

This is the Qwen3.8-27B-Uncensored Q8_0 model with its mmproj on port 9932.

## Update — 2026-08-25 evening, third pass (handoff before leaving)

### NVIDIA Q8_0 kernel: ROOT CAUSE FOUND AND FIXED

The random-data mismatch was a second CUDA warp-shuffle bug, distinct from the gater issue:

- tinygrad's `UOp.maximum` renders to a ternary `(a < shfl) ? shfl : a`, which evaluates
  `__shfl_xor_sync` **twice**. Double-evaluated shuffles return garbage (observed:
  alternating 0/32 across lanes in a max-reduce smoke test).
- Fix: `_nv_fmax(a, b)` in `tinygrad/llm/kernels/nv.py` emits a single `fmaxf({0}, {1})`
  CUSTOMI — one shuffle evaluation. Proven with `/tmp/test_nv_maxreduce2.py` → all 32.0.
- Diagnosis chain (all reproducible): `/tmp/sweep_nv_q8.py` (24/24 shapes failed before,
  all random) → stage isolation `/tmp/diag_nv_stage.py` (xq words wrong, raw+scale+decode
  correct) → `/tmp/test_nv_maxreduce.py` (max-reduce broken, sum-reduce fine) →
  `/tmp/dump_max_src.py`/DEBUG=4 source dump showing the ternary.

**Status after fix: `python3 /tmp/sweep_nv_q8.py` passes 24/24 with maxerr=0** (out 1-32,
in 32-128, all groups, all chunks). Kernel math is exact for Q8_0.

### In-flight (launched detached via setsid — survives session end)

Both on node-lair, logs under /tmp, `pgrep -af` to check:

1. **Real-shape sweep**: `python3 /tmp/sweep_nv_real.py` -> `/tmp/sweep_nv_real.log`
   (PIDs 2948930/2948932). Tests (10240,5120), (5120,17408), (17408,5120,tok=2),
   (151936,1024) — the chunked `result.sum(-1)` path and multi-token. Expect all OK;
   if any FAIL, the chunk-sum path or multi-token indexing is wrong.
2. **Beam cache training**: `CACHEDB=/tmp/tg-beam-cache.db BEAM_CACHE=1 BEAM=2
   python3 -m tinygrad.llm --model qwen3.5:0.8b --max_context 512 --benchmark 4`
   -> `/tmp/tg_cache_train.log` (PIDs 2948813/2948814). ~4-6 min.

### Beam cache patch status

- `tinygrad/codegen/opt/postrange.py` modified: `_beam_cache_key()` + diskcache
  get/put in `apply_opts`, gated on `BEAM_CACHE=1` env. Key = normalized AST key +
  renderer target + BEAM_ESTIMATE + TF32.
- Training now running detached. **Verify after it finishes:**
  - DB rows: `python3 -c "import sqlite3;print(sqlite3.connect('/tmp/tg-beam-cache.db').execute('SELECT count(*) FROM beam_opts_22').fetchone())"` (expect 60+)
  - Replay: `CACHEDB=/tmp/tg-beam-cache.db BEAM_CACHE=1 DEBUG=3 python3 -m tinygrad.llm
    --model qwen3.5:0.8b --max_context 512 --benchmark 4` (no BEAM flag)
  - Expect `beam cache hit` lines at DEBUG>=3 and ~3.5-4 ms/token without search time.

### Next steps after the two logs are green

1. Integrate: in `tinygrad/llm/kernels/amd.py::Linear` add Q8_0 support —
   `set_quantized` needs a Q8_0 branch (34 bytes/block -> uint16 view, 17 words; the
   `packed_sizes` dict lookup needs the right byte count) and `__call__` must route
   `ggml_type == 8` on NV/CUDA to `tinygrad.llm.kernels.nv.q8_0_linear`. AMD path and
   generic fallback unchanged. Note `nv.py` reads words as `uint16`; AMD uses `uint32`
   views — keep the two formats separate.
2. Proxy verify: greedy logits/tokens `qwen3.5:0.8b` must match the generic path
   (small tolerance — activation quantization is lossy vs the reference path).
3. Benchmark 27B (needs exclusive GPU: pause llama-server first — see service notes;
   restart after). Compare against 2.06 tok/s baseline.

### Files/artifacts added this pass

- `/tmp/sweep_nv_q8.py` — 24-case exactness sweep (passing)
- `/tmp/sweep_nv_real.py` — realistic-shape sweep (in flight)
- `/tmp/diag_nv_stage.py` — stage isolation (xq vs raw vs decode)
- `/tmp/test_nv_maxreduce.py` / `/tmp/test_nv_maxreduce2.py` — max-reduce repro/fix
- `/tmp/dump_max_src.py` — in-process kernel source dump
- Local mirrors: `/Users/deven367/tmp/tinygrad-work/`, `/Users/deven367/tmp/sweep_nv_q8.py`
- llama-server still healthy on :9932 (PID 2932371)

## Update — 2026-08-26 night, Q4_K session part 1

Kernel `tinygrad/llm/kernels/nv_q4k.py` VERIFIED exact (sweep 8/8, rel<=1.8e-06 incl.
chunks>1 + partial warp). All prior failures were sweep-harness bugs: sequential-vs-
superblock nibble layout, mis-packed scale bytes (get_scale_min_k4 truth documented in
sweep_q4k.py header), stale `-8` in reference, non-exact activations (pin group max
+-127), missing per-output-row tiling. amd.py wired: Q4_K->nv_q4k.q4_k_linear on NV +
device-gated set_quantized (also fixes pre-existing stock crash: K-quants on NV packed
then hit generic matmul with flat weight -> transpose IndexError). Q8_0 sweep regression
24/24 maxerr=0. Everything UNCOMMITTED on qwen27b-nv-q8-kernel. Gotchas: run WITHOUT
DEV env (Device.DEFAULT=NV; scripts' "NVK:" prefix defeats nv_custom_kernels_supported);
detection is lazy (post-forward counts only). REMAINING BLOCKER: set_quantized finds no
uint8 SHRINK candidate for Q4_K_M tensors on NV -> ggml_type stays None -> A/B asserts.
Diff weight.uop graphs 0.8b(Q8_0, works) vs 4b(Q4_K_M) next. Full details:
HANDOFF_Q4K.md. Scripts: /u/demistry/sweep_q4k.py, /u/demistry/proxy_ab_q4k.py.

## Update — 2026-08-26, Q4_K session part 2: BLOCKER RESOLVED, A/B GREEN

The "set_quantized finds no Q4_K SHRINK" blocker was a broken test harness, not the
wiring. proxy_ab_q4k.py::count_type walked dicts/__dict__ but NOT lists; Transformer.blk
is a plain list, so block Linears were never counted and only .output (Q6_K in Q4_K_M
recipes -> correctly unclaimed) was seen -> n_q4 always 0. proxy_ab.py (Q8_0) had list
handling; the Q4_K copy lost it. Fixed (list/tuple walk) -> **A/B GREEN 32/32 token
positions** on qwen3.5:4b Q4_K_M (temp 0): CUSTOM q4_k_linears=131, GENERIC=0.
Detection breakdown on 4b (249 Linears): 131 Q4_K + 48 Q8_0 claimed by set_quantized,
70 unclaimed = Q6_K weights (output + some ffn) -> generic path, correct behavior.
Diagnostic diag_q4k_graph.py walks incl. lists and calls set_quantized directly.

Committed & pushed to fork deven367/tinygrad branch qwen27b-nv-q8-kernel (identity
deven367 <masterdeven@gmail.com> via ~/bin/git-personal):
  ad94c9619 add NVIDIA Q4_K custom linear kernel (verified exact: sweep 8/8, proxy 32/32)
  2d46ea489 route Q4_K on NVIDIA to the custom nv q4_k_linear kernel; gate set_quantized claims by device
Note: commit identity fixed retroactively on all branch commits (was "Deven Mahesh Mistry
<demistry@lair-*>") via filter-branch + force-push; agent-handoffs repo likewise.

IN PROGRESS: 27B Q4_K_M benchmark. File was missing on node-lair; downloading
mradermacher/Qwen3.8-27B-OBLITERATED-GGUF Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf (16.8 GB)
to /scratch/local/demistry/models/. Custom run: python3 -m tinygrad.llm --model ...Q4_K_M.gguf
--max_context 512 --benchmark 20. Generic baseline same file: /u/demistry/bench_generic.py
(sets amd.Linear.use_custom_quant=False). GPU free tonight (nvidia-smi 0 MiB).
