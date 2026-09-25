# Handoff: Lever A REJECTED (2026-09-25, later session)

> **Outcome: Lever A (compact GEMV output buffers) is dead.** The design was implemented
> exactly as specified, broke parity, and was reverted. Root cause is in tinygrad's CUDA
> codegen for a *lane-invariant* store. The premise it rested on — "up to 572 contiguous-copy
> kernels per step" — **does not exist**: the decode graph contains zero `copy` kernels.
> This handoff supersedes `handoff-2026-09-25-lever-a-compact-gemv-in-progress.md` (kept as
> the design record).

**tinygrad (g37)**: `38342a3be`, branch `qwen27b-nv-q8-kernel`, **clean tree — patch reverted.**
**Server**: llama-server was DOWN all session (nothing on :9932, GPU 0 MiB), so every number
below is a clean measurement. `make serve` at the end of the session.

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

`DEBUG=5` dump of the patched `nv_linear_q6_k_v2` (Q6_K v2 kernel, PTX lines 300-313):

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

**Reusable rule:** on this tinygrad fork, a custom-kernel store may only be written from an
address that depends on the lane axis. The one working duplicate-value store
(`_q8_quantize_kernel`'s `scale[token, group, 0]`) survives only because the same kernel also
has a lane-dependent store (`q[token, group, word_lane]`) that pins the structure.

**Bandwidth-only variant, if anyone revisits it:** keep the address lane-dependent
(`out[token, output, lane & 3]` into a `(rows, 8)` buffer) — that survives codegen but only
saves 96 B of the 128 B/row, and the whole prize is small (see §3). Not worth it before §4.

## 3. Why it was never worth 572 kernels — the copy kernels do not exist

`JIT=0 DEBUG=2` census of one decode step at `38342a3be`, `*** CUDA` lines tallied by family
(**zero `copy` entries**): the consumer's `Tensor(out.after(kernel))[..., 0]` is folded into
the consumer as strided index arithmetic, not a standalone kernel. So the "eliminate up to 572
contiguous-copy kernels" premise — the entire justification for Lever A — was an assumption
that a 10-minute census disproves. The only real prize was the write/read bandwidth:
2.4 M rows × 128 B ≈ 300 MB/token written + the same gathered back ≈ **~0.18 ms/tok (1.3 %)**.

**Lesson: run the census before designing the kernel.** It is cheaper than the kernel.

## 4. Fresh GPU-time ranking (`38342a3be`, H100, clean)

Decode graph = **1,588 kernels/step**. Sum of `tm` over the final 1,588 launches (ranking only —
DEBUG=2 syncs per kernel, so absolutes run ~1.3-1.5× the 13.54 ms wall):

| family | GPU time (window) | calls | note |
|---|---:|---:|---|
| `nv_linear_q4_k_v4` | 6.03 ms | 318 | GEMV, weight-streaming bound |
| `nv_linear_q6_k_v2` | 1.85 ms | 43 | GEMV |
| **`r_2_32_4_970`** | **1.52 ms** | **2** | **~760 µs per call — see §5** |
| `nv_q8_quantize` | 1.50 ms | 188 | |
| `E_40_32_4` | 1.33 ms | 175 | elementwise |
| `nv_rmsnorm` | 0.99 ms | 106 | |
| `E_64_32_3` / `E_136_32_4` / `E_16_32_4` | 0.80 / 0.74 / 0.68 ms | 104 / 94 / 68 | elementwise |
| `nv_add_rmsnorm` | 0.64 ms | 47 | |
| `E_3_4_4` / `nv_normalize` / `E_16_32_4_3` | 0.60 / 0.55 / 0.51 ms | 68 / 69 / 70 | |
| `gated_delta_prefill` / `flash_decode_partial` | 0.36 / 0.35 ms | 35 / 12 | attention/recurrence |
| `copy` | **0** | **0** | ← the Lever A premise |

## 5. Ranked next steps (replaces the old list)

1. **Identify and fix `r_2_32_4_970` (~1.5 ms/step, 11 % of the step).** It is the last kernel
   pair of every decode step and its tensor is 248,320 = 2·32·4·**970** = the vocabulary size,
   so it is the logits argmax/sample stage (~760 µs to reduce 1 MB ≈ 1.3 GB/s, ~50× slower than
   it should be). llama.cpp does the same argmax in microseconds. Reproduce: run a 2-step
   decode with `DEBUG=5` and take the source block matching the kernel's ordinal in the
   `*** CUDA` sequence; then decide between a fused single-pass argmax kernel and fixing the
   reduce's launch geometry. If it can be cut to ~50 µs/call, that alone closes ~70 % of the
   1.9 ms gap to llama.cpp.
2. **Fuse the remaining elementwise families** (`E_40_32_4`, `E_64_32_3`, `E_136_32_4`,
   `E_16_32_4`, `E_3_4_4`, `E_16_32_4_3` ≈ 3.9 ms/window). SwiGLU + RoPE + the second residual
   add are still separate elementwise kernels; the 2nd residual add can ride into the next
   block's `attn_norm` (cross-block `Transformer.forward` restructure).
3. **GEMV streaming efficiency**: ~7.9 ms of GEMV per step against a hard floor of
   16.8 GB/token ÷ 3.35 TB/s = 5.0 ms. That is where the rest of the gap lives, but it is
   deep work — only worth it after 1 and 2.
4. Multi-warp grid-fused RMSNorm + Q8 quantize (old TODO 3) stays queued behind 1-2.

## 6. Commands (regenerate everything above)

```bash
ssh g37
cd ~/projects/agent-handoffs
make bench-tg        # 73.83 tok/s (13.54 ms) clean at 38342a3be — keep the server DOWN
# kernel census + per-family GPU time (writes ~3.8 MB log; node-local, not a transfer path):
JIT=0 DEBUG=2 PYTHONPATH=$HOME/projects/tinygrad-src DEV=CUDA \
  CUDA_PATH=/N/soft/rhel8/cuda/12.6/targets/x86_64-linux \
  MODEL=/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf \
  python3 qwen3.8-27b-tinygrad/scripts/bench_decode.py 512 3 > $HOME/scratch/leverA_before.log 2>&1
grep -a "^\*\*\* CUDA" $HOME/scratch/leverA_before.log | tail -1588 \
  | awk '{n=$4; for(i=1;i<=NF;i++) if($i=="tm"){v=$(i+1); sub(/\/.*/,"",v); sub(/us/,"",v); s[n]+=v; c[n]++}} END{for(k in s) printf "%9.1f us %5d %s\n", s[k], c[k], k}' | sort -rn
# store-codegen check for any custom kernel: DEBUG=5 on scripts/sweep_q6k.py, then
tr -d '\000' < log | grep -n "st.global" -B12
```

Slurm: `ssh g37` is Slurm-gated. `scontrol requeue` **does not work on interactive jobs**
("Only batch jobs are accepted") and `scontrol update JobId=… TimeLimit=…` is denied
("Access/permission denied") — the 1 h limit is hard; start a fresh `salloc` on
`h100-debu`/`interact` instead of trying to extend.
