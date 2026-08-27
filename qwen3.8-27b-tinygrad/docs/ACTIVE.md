# ACTIVE — Current state & next steps (read FIRST)

> Start here. Everything else in `sessions/` is historical context.
> Snapshot: 2026-08-27 (after NaN fix).

## Ground truth right now

**Working, verified:**
- Full custom-kernel logits on **Unsloth `UD-Q4_K_M`** are **finite and argmax-identical to generic** (argmax 5328, corr 0.99989). This unblocks the original ask: run the Unsloth quant through tinygrad fast.
- Loaders exact for `Q3_K` and `IQ4_NL` (random + real GGUF blocks, `rel ≤ 1e-8`).
- NVIDIA GEMV kernels verified exact vs numpy reference: `Q8_0`, `Q4_K`, `Q6_K`, **`Q5_K`** (new), **`IQ4_XS`** (new). All sweeps pass.
- Fixed root cause of earlier NaNs: `IQ4_NL` (18 B/32 elems) collided with `Q4_K` (144 B/256) in the byte-count claim table → 7 tensors misrouted. Fix: loader threads real GGUF types (`tinygrad.tensor_types`) and `from_gguf` stamps `Linear.ggml_type` (see session 06).

**Files changed (uncommitted on node-lair, branch `qwen27b-nv-q8-kernel`):**
```text
 M tinygrad/llm/gguf.py            # tensor_types side-channel
 M tinygrad/llm/model.py           # stamp Linear.ggml_type + _needs_pack
 M tinygrad/llm/kernels/amd.py     # type-guided claim + _needs_pack gate
?? tinygrad/llm/kernels/nv_q5k.py  # new Q5_K GEMV
?? tinygrad/llm/kernels/nv_iq4xs.py# new IQ4_XS GEMV
```

**Machine:** `node-lair` (L40S). Worktree `/u/demistry/tinygrad-src`. GPU was clean at last handoff.

## Next actions (prioritized)

### 1. Commit the fixed tree (blocks everything)
```bash
ssh node-lair
cd /u/demistry/tinygrad-src
git status   # expect the 5 files above
# commit loader type threading + stamping + new kernels as logical commits,
# author: deven367 <masterdeven@gmail.com> (use ~/bin/git-personal wrapper)
```

### 2. Complete Unsloth token A/B (was canceled mid-run)
```bash
python3 /u/demistry/p1/ud_ab.py custom /scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf 12
python3 /u/demistry/p1/ud_ab.py generic /scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf 12
diff /tmp/ab_custom.txt /tmp/ab_generic.txt   # expect identical
```

### 3. Benchmark Unsloth file with custom kernels
```bash
python3 -m tinygrad.llm --model .../Qwen3.8-27B-UD-Q4_K_M.gguf --max_context 512 --benchmark 15
```
- Read the **decode graph line** (`*** NV ... tm Xms`), not the prefill-dominated tok/s.
- Compare to `28.5 tok/s` historical (different file: Uncensored Q4_K_M) — this file has Q5_K/IQ4_XS custom now.

### 4. llama.cpp same-file comparison (completes ask)
- `llama-bench -m ...UD-Q4_K_M.gguf -p 512 -n 128` (non-interactive; interactive `llama-cli` hung).
- Compare `tg` (decode) and `pp` (prefill) on the same GPU/file.

### 5. Validate chunked prefill gate removal (`29a306ec6`)
- Compares `chunk_size=32` vs forced `chunk_size=1` token-identical output, then prefill ms. Restore the gate if regression.

### 6. (Later) DeltaNet / BEAM_CACHE / MTP — see plan P1/P5/P6 in progress.md.

## How to read this tree

```
docs/
  ACTIVE.md            <- THIS FILE. Start here every session.
  progress.md          <- timeline of all sessions + results (append here)
  kernels-explained.md <- how the custom kernels work (reference)
  sessions/            <- frozen session notes, numbered; read for context/evidence
    01 P0 profile + Q6 plan
    02 Q6_K build
    03 Q6_K verified; Unsloth switch
    04 corrections + Unsloth plan
    06 NaN FIXED (root cause + diff) — most relevant one for current work
```