# ACTIVE — Current state and next steps

> Snapshot: 2026-09-06. Priority 1 is Q8_K_XL at native 262K on one L40S.

## Read first

1. `sessions/08-2026-09-06-q8-k-xl-l40s-plan.md` — **primary plan:** quantized KV so Q8_K_XL fits at 262K.
2. `sessions/09-2026-09-06-q4-262k-prefill-plan.md` — **secondary plan:** close the 88× tinygrad prefill gap on the already-fitting Q4 control.
3. `sessions/07-2026-09-05-262k-context-plan.md` — superseded umbrella plan; retain for investigation history.

## Ground truth

- Host: `node-lair`, one L40S with `46068 MiB` / `44.99 GiB`.
- tinygrad: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`.
- Source is clean and pushed at `1d8b8cab7 fix(llm): enforce prompt+completion context boundary`.
- Launcher: `/u/demistry/agent-handoffs/Makefile`; `~/Makefile` symlinks to it.
- GPU was clean at the last check.

## Primary objective

Serve this exact model at its native `262144` context on the L40S:

```text
/scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
```

Current tinygrad uses FP16 KV. Exact full-context cache sizes:

| KV format | Size at 262K | Projected Q8 server total |
|---|---:|---:|
| FP16 | 16.0 GiB | >45 GiB; does not fit |
| Q8_0 | 8.5 GiB | ~41.8 GiB; narrow headroom |
| Q4_0 | 4.5 GiB | ~37.8 GiB; production target |

llama.cpp already selects Q4_0 KV on the L40S through `--cache-type-k/--cache-type-v`. tinygrad has no quantized-cache attention path. Implement Q8_0 first as the simpler correctness milestone, then Q4_0 for llama.cpp-equivalent headroom.

## Safety warning

The tracked Makefile currently defaults to:

```make
TG_MODEL = Qwen3.8-27B-UD-Q8_K_XL.gguf
TG_CTX   = 262144
```

That combination is a known OOM with FP16 KV. Plain `make serve-tg` is not a valid readiness check until quantized KV lands. Priority-one Phase 0 makes the launcher fail safe.

## Verified completed work

- Q8_K_XL warmup/generation at context 512: ~29.5 GiB tracked.
- Q8 server at context 4096: `34316 MiB` by `nvidia-smi`; API request passed.
- Q4_K_M full context 262144: warmup passed; server used `36904 MiB` / 36.04 GiB.
- Prompt-plus-requested-completion overflow is rejected by `serve.py`.
- Q4_K_M apples-to-apples L40S benchmark:

| Engine | KV | pp512 | decode |
|---|---|---:|---:|
| llama.cpp | FP16 | 2532.5 tok/s | 38.86 tok/s |
| llama.cpp | Q4_0 | 2503.4 tok/s | 38.49 tok/s |
| tinygrad | FP16 | 28.5 tok/s | ~28.7 tok/s |

The llama.cpp rows used three repetitions; tinygrad pp512 used one. llama.cpp VRAM in the old table was estimated. llama-bench did not enable MTP.

## Immediate next action

Execute Phase 0 then Phase 1 of session 08:

1. Make the current launcher fail safe while Q8 262K is unavailable.
2. Add explicit `f16|q8_0|q4_0` cache-format plumbing with `f16` library default.
3. Implement compact append-only Q8_0 cache storage.
4. Implement NVIDIA tiled decode attention that consumes the packed cache directly.
5. Validate at context 4096 before attempting 262144.

Keep recurrent `chunk_size=1` until the cache path passes the full-context memory gate. Prefill optimization is session 09 and remains second priority.

## Important paths

```text
Q8:       /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
Q4:       /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
tinygrad: /u/demistry/tinygrad-src
launcher: /u/demistry/agent-handoffs/Makefile
logs:     /tmp/tinygrad-server.log
```

The old `/scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf` fixture is absent. Recent tinygrad benchmark execution used the direct `NV` backend successfully; do not retain the older blanket `NV: EPERM` claim without reproducing it.
