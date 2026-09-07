# Handoff — chunked-prefill correctness: revert done, divergence localized to blk00

Date: 2026-09-07. Prior docs: `handoff-2026-09-06-chunked-prefill-codegen.md` (still
valid for background + repro commands), `docs/progress.md`, `docs/ACTIVE.md`.

## TL;DR for the next agent

- The previous agent's `t % T_actual` modulo "fix" was **actively corrupting**
  `recurrent_state` (re-reads real token rows in padded scan steps). It is **reverted**
  (`8dbe2d1ef`, pushed). Do NOT reintroduce it.
- What remains, in order: (1) keep the two good fixes, (2) find why partial-chunk
  prefill diverges starting inside **block 0's GatedDeltaNet recurrent path**.
  A working bisection script exists: `scripts/bisect_blocks.py`.

## Exact persisted state (all pushed — no /tmp files anywhere)

tinygrad-src (`/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`, fork
`deven367/tinygrad`, HEAD pushed):

| commit | file:line | what | verdict |
|---|---|---|---|
| `39a790966` | `tinygrad/renderer/cstyle.py:253-257` | emit closing braces when END uops dropped, `depth > 1` | KEEP (harmless safety net) |
| `12fa30c29` (part) | `tinygrad/runtime/graph/cuda.py:20-22` | `{v:1}` + `max(1,x)` clamp on launch dims | KEEP but note: **dead code on NV** — `ops_nv.py` wires no graph runner; only `ops_cuda.py:120-121` uses `CUDAGraph`. Makefile default is `TG_DEV ?= NV` |
| `12fa30c29` (part) | `tinygrad/llm/model.py` guard removal (old line 534) | `chunk_size=1` gate deleted | KEEP (conditional on correctness fix below) |
| `8dbe2d1ef` (HEAD) | `tinygrad/llm/model.py:318,365-369` | revert of modulo; plain `t` loop + original `T_pad = x.max_shape[1]` restored | KEEP |

Working tree clean except: `git status --short` on node-lair shows nothing
(`cuda.py.bak` removed during review). Verify with `git -C /u/demistry/tinygrad-src status --short`.

agent-handoffs (`/u/demistry/agent-handoffs`, = origin `main` after pull):
`scripts/bisect_blocks.py` (`9688c22`) — drives the REAL `__call__`/TinyJit path,
hooks `Transformer.forward` at class level (must patch BEFORE `from_gguf`, since
`TinyJit(self.forward)` binds at `__init__`), shrinks stashed symbolic block
outputs to bound `n_toks` before `.numpy()` stats.

## Key evidence (fresh process per run, `DEV=CUDA`, Q4_K_M, `max_context=512`)

Bisection output (real JIT path, mean/std/min/max per block, rel = max relative
stat difference, tol=1e-5):

```text
# partial chunk: [1], cs=1 -> tok 271 vs cs=2 -> tok 1
DIVERGE blk00 GatedDeltaNetBlock: a_mean=+2.400039e-03 b_mean=+2.395725e-03 rel=1.80e-03
# full chunk: [1,2], cs=1 -> tok 271 vs cs=2 -> tok 220
DIVERGE blk00 GatedDeltaNetBlock: a_mean=+2.400039e-03 b_mean=+2.687492e-03 rel=1.07e-01
```

Read: divergence starts at **blk00 itself** (small on `[1]`, large on `[1,2]` —
scales with real-token content in the window). Not downstream accumulation, not
the final `x[:, -1:]` select (`model.py:407`). The full-chunk `[1,2]` cs=1-vs-2
match the old agent reported (220==220) was stale/JIT-state-contaminated; fresh
processes diverge there too.

Repro (from repo root on node-lair):

```bash
git -C /u/demistry/agent-handoffs pull --ff-only
DEV=CUDA python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bisect_blocks.py "[1]" 1 2 6
DEV=CUDA python3 /u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/scripts/bisect_blocks.py "[1,2]" 1 2 6
```

## Suspect list inside blk00 (`GatedDeltaNetBlock._attention`, `model.py:311-378`)

1. **Conv window + stored `conv_state`** (`model.py:331-342`): `win` is static
   `conv_kernel-1 + T_pad` wide; real rows stored at
   `[conv_kernel-1 : conv_kernel-1+T]`; padded tail is zeros per the "exact
   no-op" comment. Next-state slice `conv_window[:, T:T+conv_kernel-1]` uses
   symbolic `T`. Question: does the bound `toks` value flow correctly into both
   the window store width AND the next-state slice when `n_toks < chunk_size`?
2. **`pad_to(T_pad)` on gate/beta/log_alpha** (`model.py:345-346`): padding
   values (zeros) become `alpha=exp(0)=1`, `beta=sigmoid(0)=0.5` downstream
   after the transforms — is `beta=0.5` on padded steps really a no-op in the
   delta rule, or does the "beta=0" comment describe a different tensor?
3. **Stored `recurrent_state` write** (`model.py:375`): the scan writes back the
   post-`T_pad`-step state including padded steps. If padded steps are not
   exact no-ops, the stored state is poisoned for the next chunk.
4. `forward`'s `x[:, -1:]` (`model.py:407`): ruled OUT as the origin (blk00
   already diverges), but still relevant for output slicing once blk00 is fixed.

## Suggested next steps (in order)

1. Intra-blk00 instrumentation: extend the `bisect_blocks.py` hook pattern to
   stash `conv_out`, `q/k/v`, `beta/alpha`, and scan `core` inside `_attention`
   (hook `GatedDeltaNetBlock._attention` at class level the same way). Compare
   cs=1 vs cs=2 on prompt `[1]`; the first diverging intermediate names the
   subsystem (conv vs gate/beta/alpha vs scan-core).
2. If padded-step no-op-ness is the culprit, the correct construction is a
   validity gate (`valid = t < T_actual; state = valid.where(new_state, state)`,
   outputs similarly gated) — NOT modulo indexing. Alternatively, handoff
   Approach A: pad inputs to exactly `chunk_size` in `generate()` (`model.py:533-552`).
3. Only then: A/B greedy tokens cs=1 vs 2/4/8 on several prompts, then prefill
   throughput measurement. Watch `chunk_size=32` VRAM (symbolic GEMV scratch;
   see `handoff-2026-09-05-vram-oom.md`).
4. Do NOT touch: `cstyle.py` brace fix, `cuda.py` launch-dims fix, guard removal.

## Gotchas learned this session (node-lair specifics)

- Remote shell is **zsh**: `echo ===` breaks scripts; avoid `===` separators in
  `ssh` commands. Quote carefully; heredocs via `cat > file << "ENDSCRIPT"`
  (quoted delimiter) survive; unquoted delimiters with `$`/backticks do not.
- **Never use /tmp for scripts** — the allocated compute node changes between
  sessions and /tmp does not persist. All scripts live in agent-handoffs
  (`scripts/`) and are pulled on node-lair; all source fixes are committed +
  pushed (tinygrad-src → `fork/qwen27b-nv-q8-kernel`, agent-handoffs → `origin/main`).
- Every python probe must be a **fresh process** (fresh model load): reusing a
  process across `generate()` calls contaminates recurrent/KV caches and JIT
  state and produces false "match" results.
- `block(x, sp).realize().numpy()` fails when the output shape is symbolic
  (`assert all_int` in `tensor.py:528`); shrink dim1 to bound `n_toks` first.
- `TinyJit(self.forward)` binds at `Transformer.__init__` — monkeypatching
  `model.forward` post-construction does NOT affect the JIT path; patch
  `Transformer.forward` at class level BEFORE `from_gguf`.
- `DEV=NV` vs `DEV=CUDA`: NV has no graph runner (`ops_nv.py` wires none);
  `runtime/graph/cuda.py` only executes under `DEV=CUDA`.
