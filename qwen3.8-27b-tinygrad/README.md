# qwen3.8-27b-tinygrad — docs & handoffs

Single home for all documentation, handoffs, and progress for the
**Qwen3.8-27B tinygrad optimization** project (node-lair, NVIDIA kernels).

## Layout

```
qwen3.8-27b-tinygrad/
  docs/                      <- ALL documentation lives here (canonical)
    progress.md                            master progress (start here)
    consolidated-handoff.md                session-5 consolidation + corrections
    p1-handoff.md                          P0 profile + P1 Q6_K build plan
    p1-q6k-handoff.md                      Q6_K kernel build + verification notes
    p1-q6k-session5-handoff.md             Q6_K verified; Unsloth switch
    handoff-2026-08-27-unsloth-optimization.md   Unsloth NaN blocker discovery
    handoff-2026-08-27-nan-fixed.md        NaN root cause FIXED (latest state)
    handoff-2026-09-17-parity-benchmark.md L40S vs llama.cpp head-to-head
    handoff-2026-09-17-root-cause-analysis.md  H100 session: why the gaps exist (START HERE for perf)
    kernels-explained.md                   human-readable kernel explanations
  kernels/    vendored kernel sources (nv.py, nv_q4k.py, nv_q6k.py, amd-routing.patch)
  scripts/    verifiation scripts (sweeps, proxy A/B)
```

## Where things live on node-lair

- tinygrad worktree: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`
- scratch scripts + logs: `/u/demistry/p1/` (scripts), `/u/demistry/logs/` (profiles)
- canonical docs are **this directory**; remote `/u/demistry/p1/HANDOFF_*` files are copies for convenience

## Reading order (recommended)

1. `docs/ACTIVE.md` — **current state + next steps (start here)**
2. `docs/progress.md` — master timeline
3. `docs/sessions/` — frozen historical handoffs (numbered; read for context)
4. `docs/kernels-explained.md` — how the kernels work

## Status (one line)
**See `docs/ACTIVE.md`** for the current state and next actions.