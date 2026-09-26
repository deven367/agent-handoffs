# HANDOFF — next agent (2026-09-25, end of session)

**Read this, then `handoff-2026-09-25-lever-a-rejected-and-reprofile.md` for the full evidence.**

## State you inherit

| item | state |
|---|---|
| tinygrad (g37) | `38342a3be`, branch `qwen27b-nv-q8-kernel`, **clean tree — every patch from this session was reverted**, pushed, `origin` in sync |
| agent-handoffs (g37) | **`6edbde5`, needs a `git pull`** — the Mac repo has this handoff and `scripts/greedy_token_ab.py` pushed on top of `6edbde5` (`git pull` before you need the probe script) |
| llama-server | **DOWN** (nothing on :9932, GPU 0 MiB). I lost node access before I could run `make serve` — restore it, see §1 |
| Slurm | **no allocation.** I cancelled the interactive job I was ssh'd into (which also killed my shell) and had queued `10644117` (`h100-debug`, 1 h); whether it started, expired or is still pending is unknown. `ssh g37` fails with `you have no active jobs on this node (pam_slurm_adopt)` until a job holds the node — submit one, §3 |
| perf | tinygrad **73.83 tok/s (13.54 ms)**, llama.cpp **85.99 ± 1.43 tok/s (11.66 ms)**, both re-measured clean this session → **85.9 % parity, 1.91 ms/tok gap** |

## 0. First 5 minutes

```bash
ssh g37          # if this fails with "you have no active jobs on this node" (pam_slurm_adopt),
                 # you have no allocation: submit one and retry
squeue -u demistry -o "%.10i %.9T %.10M %.10l %j"     # see what is running/pending
nvidia-smi --query-gpu=memory.used --format=csv,noheader
cd ~/projects/agent-handoffs && git pull && git log --oneline -1   # expect 6edbde5 or newer
cd ~/projects/tinygrad-src && git status --short && git log --oneline -1   # expect clean, 38342a3be
make bench-tg      # must print ~73.83 tok/s (13.54 ms) — re-establish the baseline before changing anything
```

## 1. Restore the user's llama-server

It is the user's server (port 9932, Q8_K_XL + MTP, ~44 GiB) and it was down the whole session.
**Benchmark hygiene: a resident llama-server cut llama-bench 85.87 → 48.51 tok/s (44 % SM
bursts), so keep it DOWN while benchmarking and start it when you are done.**

```bash
cd ~/projects/agent-handoffs
make serve     # nohup + health probe; then confirm it survives >30 s:
make status; sleep 35; make status; nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

The server lives inside the Slurm allocation — when the allocation ends, the server dies with
it (that is almost certainly what happened at 15:09 earlier today; the log's
`cleaning up before exit` at `6:10` elapsed is a normal shutdown, not a crash). Do not spend
time debugging a "server crash" that coincides with a job ending.

## 2. The job: a custom greedy-argmax kernel (the whole remaining gap, basically)

**Where the time goes** (census at `38342a3be`, decode graph = 1,588 kernels/step, 13.54 ms):

| family | GPU time / step |
|---|---:|
| GEMV (`nv_linear_q4_k_v4` + `nv_linear_q6_k_v2`) | ~7.9 ms (floor: 16.8 GB ÷ 3.35 TB/s = 5.0 ms) |
| **`r_2_32_4_970` — the vocab argmax, 2 calls** | **~1.5 ms** |
| `nv_q8_quantize` | ~1.5 ms |
| norms (`nv_rmsnorm`, `nv_add_rmsnorm`, `nv_normalize`) | ~2.2 ms |
| elementwise `E_*` | ~3.9 ms |
| attention/recurrence | ~0.7 ms |

`r_2_32_4_970` reduces the 248,320-entry vocab with **`blockIdx.x /* 2 */` × `threadIdx.x
/* 32 */` = 64 threads total** and a 970-iteration loop, with the Gumbel correction fused in
and a 7,946,240-float (32× logits) intermediate. tinygrad's generic reduce scheduler picks
this; it is latency-bound, not bandwidth-bound. The data is 1 MB — this should take tens of
microseconds. **Fixing it is worth ~1.4 ms/token ≈ 73 % of the remaining 1.91 ms gap.**

### What already failed — do not repeat these

| attempt | outcome |
|---|---|
| `noisy.reshape(b,-1,32).argmax(-1).argmax(-1)` | **silently WRONG tokens.** `argmax` of per-group argmax *indices* is an argmax over 0..31, not over logits — it returns a row/column number, not a vocab id. Its 77.62 tok/s was a fake win |
| `groups.argmax(-1)` + `groups.max(-1)` + `row*32 + gidx.gather(1,row)` | provably correct (matches numpy + `argmax` incl. ties) but **1.65 ms slower** (62.58 tok/s) — two full-tensor reduces instead of one |
| Lever A compact GEMV output buffers (`(rows,32)` → `(rows,1)`) | **rejected**: parity broke, and the copy kernels it targeted do not exist. Full analysis in the rejection handoff |

### What to build

Two custom kernels in `tinygrad/llm/kernels/` (same style as `nv.py`, reuse
`_warp_reduce`/`_nv_shuffle_xor`/`_nv_fmax` and the `nv_custom_kernels_supported` guard for a
generic fallback):

1. `nv_argmax_partial(x: (tokens, vocab) f32) -> (pv, pi)`, one **warp per 32-element chunk**
   (7,760 warps for a 248,320 vocab):
   - `vmax = _warp_reduce(v, maximum=True)` (5 × `shfl_xor` + `fmaxf`)
   - `winner = (v == vmax) ? lane : 32`, then a warp **min**-reduce → `idx = chunk*32 + imin`
   - store `pv[t, chunk] = vmax`, `pi[t, chunk] = idx`
2. `nv_argmax_final(pv, pi) -> (1,) i32`: a single block, 32 threads, loop over 7,760
   partials keeping `(best_val, best_idx)`; tie rule = **lowest flat index wins** (this is what
   tinygrad's `argmax` does — verified on a constructed tie).

In `Transformer.forward`, keep the Gumbel math exactly as is and replace only the final
`.argmax(-1, keepdim=True)` on `noisy` with the two-stage path, falling back to
`noisy.argmax(-1, keepdim=True)` when the device is not NV/CUDA.

Cost target: 1 MB read + ~31 KB partials → tens of microseconds, i.e. 13.54 → ~12.1 ms/tok
≈ **82-84 tok/s**. It can be fused into the lm_head GEMV later; don't attempt that now.

### Verification (in this order — each gate is cheap and catches a different class of bug)

```bash
# 1. unit: new sweep comparing the kernel against numpy argmax on 32000 / 128256 / 248320
#    element tensors, plus all-equal data and a constructed tie. Wire it into `make test-units`.
# 2. token identity (this is what caught the wrong-argmax attempt):
DEV=CUDA PYTHONPATH=$HOME/projects/tinygrad-src \
  MODEL=/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf \
  python3 qwen3.8-27b-tinygrad/scripts/greedy_token_ab.py
#    MUST print exactly: TOKENS [381, 310, 5790, 421, 279, 491, 2936, 1000, 381]
#    (baseline rate 69.78 tok/s; the rate is A/B-only, `make bench-tg` is the headline)
# 3. make parity      -> bit-exact golden (logits are untouched by this change, but it is 80 s)
# 4. make bench-tg    -> expect > 73.83 tok/s
# 5. re-run the census (§6 of the rejection handoff) and confirm r_2_32_4_970 is gone
# 6. commit on g37: cd ~/projects/tinygrad-src && ~/bin/git-personal   # sets identity, takes no args
#    then: git add -A && git commit && git push origin qwen27b-nv-q8-kernel
# 7. docs: update ACTIVE.md (bench table + ranked list), add a completion handoff, commit+push
#    the Mac repo (/Users/deven367/projects/agent-handoffs), then `git pull` on g37
# 8. make serve (restore the user's server) + make status
```

## 3. Ground rules that were learned the hard way today

- **Do not move files between nodes through `/tmp`** — it is per-node. Anything another node
  or the next session needs goes into the **agent-handoffs repo and gets pushed to GitHub**
  (that is why `scripts/greedy_token_ab.py` exists in the repo rather than in a scratch dir).
- **A custom-kernel store must go through a lane-dependent address.** A lane-invariant store
  makes the CUDA renderer gate the store and re-emit the last `shfl_xor` butterfly, so the
  value stored is doubled (see the PTX excerpt in the rejection handoff §2).
- **`make parity` does not test the sampler.** Only `greedy_token_ab.py` does.
- **Run the kernel census before designing a kernel** — the Lever A premise (572 copy kernels)
  died in ten minutes of census; the kernel itself cost an hour.
- Slurm: real partition is **`h100-debug`** (squeue truncates to `h100-debu`), submissions need
  **`-A r00117`**, **1 h is the hard max** (`-t 04:00:00` is rejected), and `scontrol requeue`
  / `scontrol update … TimeLimit` **do not work on interactive jobs**. Start a new one with
  `salloc -A r00117 -p h100-debug -N 1 -n 1 -t 01:00:00 --job-name=interact2 sleep infinity &`.
  Never `scancel` the job you are ssh'd into — it drops the connection and every command with
  it (I did exactly that at the end of this session; the queued replacement job may or may not
  have started).
- Benchmark only with the server down; `make stop` first, `make serve` after.

## 4. After the argmax kernel

1. Fuse the remaining elementwise families (~3.9 ms): SwiGLU + RoPE + the 2nd residual add
   into the next block's `attn_norm` (cross-block `Transformer.forward` restructure).
2. GEMV streaming efficiency: ~7.9 ms against a 5.0 ms HBM floor.
3. Multi-warp grid-fused RMSNorm + Q8 quantize (old TODO 3).

**Node-local artifacts from this session** (may be gone after node recycling; every one is
regenerable from §6 of the rejection handoff): `~/scratch/leverA_before.log` (kernel census),
`~/scratch/dbg5_decode.log` (5.8 MB per-kernel PTX for one decode step — this is what
identified `r_2_32_4_970`), `~/scratch/d5.txt` (same, NUL-stripped).
