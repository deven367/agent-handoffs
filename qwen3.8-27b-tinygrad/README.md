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
    kernels-explained.md                   human-readable kernel explanations
  kernels/    vendored kernel sources (nv.py, nv_q4k.py, nv_q6k.py, amd-routing.patch)
  scripts/    verifiation scripts (sweeps, proxy A/B)
```

## Where things live on node-lair

- tinygrad worktree: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`
- scratch scripts + logs: `/u/demistry/p1/` (scripts), `/u/demistry/logs/` (profiles)
- canonical docs are **this directory**; remote `/u/demistry/p1/HANDOFF_*` files are copies for convenience

## Reading order (recommended)

1. `docs/progress.md` — session timeline, results, plan
2. `docs/handoff-2026-08-27-nan-fixed.md` — current state: NaN blocker FIXED, next steps

## Status (one line)

Custom NVIDIA Q8_0/Q4_K/Q6_K (and new Q5_K/IQ4_XS) GEMV kernels; Unsloth
UD-Q4_K_M logits finite & argmax-identical to generic after the IQ4_NL≡Q4_K
byte-collision fix. Pending: token A/B confirmation, decode benchmark,
llama.cpp comparison, chunked-prefill validation, DeltaNet, MTP.