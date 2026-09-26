# Handoff: Lever A REJECTED + argmax stage identified (2026-09-25, later session)

> **Outcome 1: Lever A (compact GEMV output buffers) is dead.** The design was implemented
> exactly as specified, broke parity, and was reverted. Root cause is in tinygrad's CUDA
> codegen for a *lane-invariant* store. The premise it rested on — "up to 572 contiguous-copy
> kernels per step" — **does not exist**: the decode graph contains zero `copy` kernels.
>
> **Outcome 2: the ~1.5 ms/token vocab argmax stage is identified** (`r_2_32_4_970`, 2 blocks ×
> 32 threads). Two naive fixes were tried, measured, and reverted: one silently wrong, one
> 1.65 ms slower. It needs a custom kernel. Details + A/B harness in §5 and §7.

**tinygrad (g37)**: `38342a3be`, branch `qwen27b-nv-q8-kernel`, **clean tree — every patch
reverted.** Nothing to clean up on the node.
**Server**: llama-server was DOWN all session (nothing on :9932, GPU 0 MiB), so every number
below is a clean measurement.
**Supersedes** `handoff-2026-09-25-lever-a-compact-gemv-in-progress.md` (kept as the design
record; its §Expectation-setting and §Verification-sequence are obsolete).

## 1. What was tried

The 9 edits from the in-progress handoff, verbatim: 6 × `out[token, output, lane].store` →
`out[token, output, 0].store`, 3 × `Tensor.empty(tokens, out_features, 32, …)` → `(…, 1, …)`
in `tinygrad/llm/kernels/{nv,nv_q4k,nv_q6k}.py`.

**Result — hard failure, both gates:**

| gate | result |
|---|---|
| `make test-units` | Q4_K sweep OK; **Q6_K sweep 7/7 FAIL** (e.g. `248320x5120 t1 maxabs=173.25 scaled_rel=2.197`) |
| `make parity` | `argmax=271` but `top5=[(271, 6.3661), (478, 5.6634), …]`, `max=+6.3661`, `sum=-235396.95` — **golden is `top5=[(271, 34.0718), (25, 19.5536), …]`, `max=+34.0718`, `sum=-800723.00`** |

A/B control (patch `git stash`ed, same session, same GPU): parity reproduced golden
**bit-exact**. So the regression is the patch, not drift.

## 2. Root cause (PTX, not speculation)

`DEBUG=5` dump of the patched `nv_linear_q6_k_v2` (PTX lines 300-313):

```ptx
 300   setp.ne.s32        %p6, %r67, 0;
 301   @%p6 bra            $L__BB0_2;          // store is GATED
 303   cvta.to.global.u64  %rd20, %rd1;
 304   mov.b32             %r165, %f1;
 306   shfl.sync.bfly.b32  %r169|%p7, %r165, 1, %r156, %r157;   // extra butterfly
 308   add.f32             %f46, %f1, %f45;                       // f46 = f1 + f1
 313   st.global.f32       [%rd22], %f46;         // ...and the STORED VALUE is doubled
```

When the store's address no longer depends on `lane`, the renderer restructures the kernel:
the final reduce step is duplicated and the store is predicated. `total` is already the
all-reduce result on every lane, so the emitted `f1 + shfl_xor(f1, 1)` **double-counts**, and
the gate makes the write conditional. The in-progress handoff's claim — "the store is
unconditional, so the gater bug does not apply" — is **wrong**: making the store lane-invariant
is exactly what triggers the gater path.

**Reusable rule:** on this tinygrad fork, a custom-kernel store may only be written through an
address that depends on the lane axis. The one working duplicate-value store
(`_q8_quantize_kernel`'s `scale[token, group, 0]`) survives only because the same kernel also
has a lane-dependent store (`q[token, group, word_lane]`) that pins the structure.

**Bandwidth-only variant, if anyone revisits it:** keep the address lane-dependent
(`out[token, output, lane & 3]` into a `(rows, 8)` buffer) — that survives codegen but only
saves 96 B of the 128 B/row, and the whole prize is small (see §3). Not worth it before §5.

## 3. Why it was never worth 572 kernels — the copy kernels do not exist

`JIT=0 DEBUG=2` census of one decode step at `38342a3be`, `*** CUDA` lines tallied by family
(**zero `copy` entries**): the consumer's `Tensor(out.after(kernel))[..., 0]` is folded into
the consumer as strided index arithmetic, not a standalone kernel. So the "eliminate up to 572
contiguous-copy kernels" premise — the entire justification for Lever A — was an assumption
that a 10-minute census disproves. The only real prize was the write/read bandwidth:
2.4 M rows × 128 B ≈ 300 MB/token written + the same gathered back ≈ **~0.18 ms/tok (1.3 %)**.

**Lesson: run the census before designing the kernel.** It is cheaper than the kernel.

## 4. Fresh GPU-time ranking (`38342a3be`, H100, clean)

Decode graph = **1,588 kernels/step**; 13.54 ms/tok wall (`make bench-tg`, clean, server down).
Ranked by GPU time — sum of `tm` over the final 1,588 launches of a `JIT=0 DEBUG=2` run.
Ranking only: DEBUG=2 syncs per kernel, so absolutes run ~1.3-1.5× high.

| family | GPU time (window) | calls | note |
|---|---:|---:|---|
| `nv_linear_q4_k_v4` | 6.03 ms | 318 | GEMV, weight-streaming bound |
| `nv_linear_q6_k_v2` | 1.85 ms | 43 | GEMV |
| **`r_2_32_4_970`** | **1.52 ms** | **2** | **~760 µs per call — the vocab argmax, see §5** |
| `nv_q8_quantize` | 1.50 ms | 188 | |
| `E_40_32_4` | 1.33 ms | 175 | elementwise |
| `nv_rmsnorm` | 0.99 ms | 106 | |
| `E_64_32_3` / `E_136_32_4` / `E_16_32_4` | 0.80 / 0.74 / 0.68 ms | 104 / 94 / 68 | elementwise |
| `nv_add_rmsnorm` | 0.64 ms | 47 | |
| `E_3_4_4` / `nv_normalize` / `E_16_32_4_3` | 0.60 / 0.55 / 0.51 ms | 68 / 69 / 70 | elementwise |
| `gated_delta_prefill` / `flash_decode_partial` | 0.36 / 0.35 ms | 35 / 12 | attention/recurrence |
| **`copy`** | **0** | **0** | the Lever A premise |

## 5. Ranked next steps (replaces the old list)

1. **The vocab argmax/sample stage — ~1.5 ms/token, 11 % of the step. IDENTIFIED; two naive
   fixes tried and both FAILED; it needs a custom kernel.**
   The kernel is `r_2_32_4_970`: `__launch_bounds__(32)`, `gidx0 = blockIdx.x /* 2 */`,
   `lidx0 = threadIdx.x /* 32 */` — i.e. **2 blocks × 32 threads = 64 threads** for a
   248,320-element (vocab) reduce, looping 970 times, with the Gumbel correction
   (`log2(log2(x)*-ln2)*-ln2`) fused into it and a 7,946,240-float (32× the logits)
   intermediate. Latency-bound, not bandwidth-bound. Measurements (all on g37, all reverted):

   | attempt | tokens vs baseline | decode rate |
   |---|---|---|
   | baseline (`38342a3be`) | `[381, 310, 5790, 421, 279, 491, 2936, 1000, 381]` | 69.78 tok/s (14.33 ms) |
   | `noisy.reshape(b,-1,32).argmax(-1).argmax(-1)` | **WRONG** (`[55, 22, 3, 3, …]`) | 77.62 tok/s — fake win |
   | `groups.argmax(-1)` + `groups.max(-1)` + `row*32 + gather` | identical ✓ | **62.58 tok/s (15.98 ms) — 1.65 ms slower** |

   - Attempt 1 is not a fix: `argmax` of per-group argmax **indices** is an argmax over a
     0..31 range, not over the logits — it returns a row/column number, not a vocab id. Its
     speed is not headroom, it is a different (cheaper) computation.
   - Attempt 2 is provably correct (equal to `argmax` and to numpy on 4096/1024/32000/248,320
     elements **and** on a constructed tie, where both return the first max) but it runs *two*
     reduces over the full tensor instead of one and loses 1.65 ms.
   - **What is needed:** a custom greedy-argmax kernel in the `tinygrad/llm/kernels/` pattern
     (same style as `nv.py`): warp-per-32-chunk partial `(value, index)` across 7,760 warps,
     then a single-block second stage over 7,760 partials. 1 MB read + ~31 KB, i.e. tens of
     microseconds instead of 1.5 ms — that alone is ~70 % of the remaining 1.9 ms gap — and it
     can later be fused into the lm_head GEMV. Tie rule to reproduce: lowest index wins.
2. **Fuse the remaining elementwise families** (`E_40_32_4`, `E_64_32_3`, `E_136_32_4`,
   `E_16_32_4`, `E_3_4_4`, `E_16_32_4_3` ≈ 3.9 ms/window). SwiGLU + RoPE + the second residual
   add are still separate elementwise kernels; the 2nd residual add can ride into the next
   block's `attn_norm` (cross-block `Transformer.forward` restructure).
3. **GEMV streaming efficiency**: ~7.9 ms of GEMV per step against a hard floor of
   16.8 GB/token ÷ 3.35 TB/s = 5.0 ms. That is where the rest of the gap lives, but it is
   deep work — only worth it after 1 and 2.
4. Multi-warp grid-fused RMSNorm + Q8 quantize (old TODO 3) stays queued behind 1-3.

## 6. Commands (regenerate everything above)

```bash
ssh g37
cd ~/projects/agent-handoffs
make bench-tg     # 73.83 tok/s (13.54 ms) clean at 38342a3be — keep the server DOWN
# kernel census + per-family GPU time (~3.8 MB node-local log, not a transfer path):
JIT=0 DEBUG=2 PYTHONPATH=$HOME/projects/tinygrad-src DEV=CUDA \
  CUDA_PATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux \
  MODEL=/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf \
  python3 qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 3 > $HOME/scratch/leverA_before.log 2>&1
grep -a "^\*\*\* CUDA" $HOME/scratch/leverA_before.log | tail -1588 \
  | awk '{n=$4; for(i=1;i<=NF;i++) if($i=="tm"){v=$(i+1); sub(/\/.*/,"",v); sub(/us/,"",v); s[n]+=v; c[n]++}} END{for(k in s) printf "%9.1f us %5d %s\n", s[k], c[k], k}' | sort -rn
# store-codegen check for any custom kernel: DEBUG=5 on scripts/sweep_q6k.py, then
tr -d '\000' < log | grep -n "st.global" -B12
# per-kernel sources for a full model step (block order matches the *** CUDA ordinal):
JIT=0 DEBUG=5 python3 qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 1 > $HOME/scratch/dbg5.log 2>&1
```

**Slurm:** `ssh g37` is Slurm-gated. `scontrol requeue` **does not work on interactive jobs**
("Only batch jobs are accepted") and `scontrol update JobId=… TimeLimit=…` is denied
("Access/permission denied") — the 1 h limit is hard. The real partition name is
**`h100-debug`** (squeue truncates it to `h100-debu`), submissions need **`-A r00117`**, and
`-t 04:00:00` / `-t 06:00:00` are rejected ("Requested time limit is invalid") — 1 h is the max:

```bash
salloc -A r00117 -p h100-debug -N 1 -n 1 -t 01:00:00 --job-name=interact2 sleep infinity &
```

## 7. Greedy A/B harness (token-identical check for sampling changes)

`make parity` validates **logits** only — it does not catch a broken sampler, which is exactly
how attempt 1 above looked "fast and fine" until tokens were compared. Any change to
`Transformer.forward`'s sampling tail must be checked token-for-token:

```bash
ssh g37
cd ~/projects/tinygrad-src
export PYTHONPATH=. DEV=CUDA CUDA_PATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux \
       MODEL=/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
# script: from_gguf(max_context=512, f16) -> generate([1000], temperature=0.0)
#         -> print 9 token ids, then time 20 more steps.
# BASELINE at 38342a3be: 69.78 tok/s (14.33 ms/tok)
#   TOKENS [381, 310, 5790, 421, 279, 491, 2936, 1000, 381]
python3 probe.py      # TOKENS must match exactly; the rate is A/B-only, never a headline
```

Cheap pre-flight for any argmax rewrite — check it standalone (seconds) before spending a
95-second model run: it must agree with numpy on several vocab sizes *and* on a constructed
tie. Gotchas found the hard way: this fork's `Tensor.gather(dim, index)` takes **two**
positional args, and a staged argmax must reconstruct the flat index as `row*32 + col` —
omitting that silently yields a column number that still "looks like" a token id.
