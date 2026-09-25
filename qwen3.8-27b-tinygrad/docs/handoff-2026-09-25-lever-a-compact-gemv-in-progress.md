# Handoff: Lever A — Compact GEMV Output Buffers (IN PROGRESS, 2026-09-25)

> **State: design finalized, edits NOT yet applied.** This session landed TODO 1+2
> (`38342a3be`) and then started Lever A. Everything below is the complete work order.

**tinygrad (g37)**: `38342a3be`, branch `qwen27b-nv-q8-kernel`, clean tree, pushed.
**agent-handoffs**: `d93c556` (pushed; g37 copy synced via `git pull`).
**llama-server**: RUNNING on g37 (port 9932, Q8_K_XL + MTP, ~44.4 GiB, pid in
`/tmp/llama-server.pid`) — the user's server; **always restart it after benchmarking**
(`make serve`, then `curl -s localhost:9932/health`).

Current perf (clean, server stopped): tinygrad **73.69 tok/s (13.57 ms)**, llama.cpp
**85.87 ± 1.00 tok/s (11.65 ms)**, gap **1.92 ms/tok**, parity 85.8%.
Details: [`handoff-2026-09-25-fused-add-rmsnorm-and-compact-q8.md`](handoff-2026-09-25-fused-add-rmsnorm-and-compact-q8.md)

---

## The task

Every GEMV in `tinygrad/llm/kernels/{nv,nv_q4k,nv_q6k}.py` allocates
`out = Tensor.empty(tokens, out_features, 32, dtype=dtypes.float32)` and the kernel does
`out[token, output, lane].store(total)` — all 32 lanes of the warp write 128 B per output
row, but the consumer (`result = Tensor(out.after(kernel))[..., 0]`) reads **word 0 only**.
So ~2.4 M rows/token get 128 B written / 4 B read (~298 MB/token dead stores), and the
`[..., 0]` read is a 128 B-stride gather (8× sector amplification), and today's non-contig
stride likely forces a contiguous-copy kernel per GEMV (up to 572 copies/step).

## The fix (decided — do exactly this)

Change the store to a **duplicate same-value store from all 32 lanes into word 0**, and
shrink the buffer to `(tokens, out_features, 1)`. This is the exact pattern already proven
in `_q8_quantize_kernel`'s scale store (all lanes store the identical value to one
address — no warp predication, no gater UB, `38342a3be`):

- `total` is identical on all 32 lanes after `_warp_reduce` (xor 16/8/4/2/1 is an
  all-reduce), so 32 lanes writing the same 4 B is safe and merges into one sector.
- The store is **unconditional** (no `lane` in the address) — the nv.py gater bug
  (lane-0 *gated* store predicates the warp) does not apply.
- Consumer `[..., 0]` on a last-dim-of-1 becomes a contiguous reshape (no copy kernel).
- **Bit-exactness is trivially preserved**: same `total`, same word 0 value; only layout
  changes. Parity MUST still match golden exactly.

### Exact edit list (line numbers = current tree at `38342a3be`)

Store sites — `out[token, output, lane].store` → `out[token, output, 0].store` (6 sites):

| file | line | which kernel |
|---|---:|---|
| `tinygrad/llm/kernels/nv.py` | 185 | `_decode_linear` (q8_0) |
| `tinygrad/llm/kernels/nv_q4k.py` | 144 | `_q4_k_v4_decode_kernel` |
| `tinygrad/llm/kernels/nv_q4k.py` | 223 | `_q4_k_v2_decode_kernel` |
| `tinygrad/llm/kernels/nv_q4k.py` | 284 | `_q4_k_decode_kernel` (scalar) |
| `tinygrad/llm/kernels/nv_q6k.py` | 127 | `_q6_k_v2_decode_kernel` |
| `tinygrad/llm/kernels/nv_q6k.py` | 203 | `_q6_k_decode_kernel` (scalar) |

Allocation sites — `Tensor.empty(tokens, out_features, 32, ...)` → `(tokens, out_features, 1, ...)` (3 sites):

| file | line |
|---|---:|
| `tinygrad/llm/kernels/nv.py` | 208 (in `q8_0_linear`) |
| `tinygrad/llm/kernels/nv_q4k.py` | 293 (in `q4_k_linear`) |
| `tinygrad/llm/kernels/nv_q6k.py` | 212 (in `q6_k_linear`) |

**Do NOT touch** the three `result = Tensor(out.after(kernel))[..., 0]` lines (one per
linear function) — they work unchanged on a last-dim-of-1.

**Do NOT touch** `scripts/test_coop_q4k.py:78,86` and `scripts/test_coop_q6k.py:93,101` —
they carry self-contained copies of the old 32-word pattern in their own kernel
implementations (arithmetic-only tests); they stay green independently. Optional
hygiene: sync them to the new pattern in the same commit if you like the consistency.

## Expectation-setting (read before celebrating)

- Pure bandwidth math: ~298 MB/token writes + ~67 MB/token read amplification ≈
  **~0.15 ms/tok** at ~2.5 TB/s. The "~1 ms" in ACTIVE.md was an overestimate — the
  bandwidth part is small.
- The real win is likely the **eliminated contiguous-copy kernels** (up to 572/step) and
  their launch/graph overhead — verify with a kernel census, not by assumption.
- Measure both: `make bench-tg` AND a before/after kernel census (e.g.
  `scripts/kernel_count_compare.py` or DEBUG=2 log + kstat pattern). Baseline census:
  ~2,060 kernels/step (pre-`38342a3be`), ~1,996 expected after TODO 1+2.

## Verification sequence (in this order)

```bash
ssh g37
cd ~/projects/agent-handoffs

make test-units   # both sweeps must print ALL OK
make parity       # must be bit-exact golden:
                  #   cs=1 argmax=271
                  #   cs=1 top5=[(271, 34.0718), (25, 19.5536), (11751, 18.7959), (248044, 17.0679), (198, 16.6857)]
                  #   cs=1 n=248320 max=+34.0718 min=-12.4540 sum=-800723.00

make stop         # STOP the user's llama-server (44% SM bursts skew benches)
make bench-llama  # expect ~85.87 ± 1
make bench-tg     # expect > 73.69
make serve        # RESTART the server, confirm curl -s localhost:9932/health == {"status":"ok"}
```

Then commit on g37 (`cd ~/projects/tinygrad-src; ~/bin/git-personal` — **the wrapper sets
identity but does NOT forward args**, then plain `git add -A && git commit`), message
suggestion: `perf(nv): compact GEMV output buffers (4 B/row instead of 128 B)`, push
(`git push origin qwen27b-nv-q8-kernel`), update `docs/ACTIVE.md` (bench table, census,
next-steps — **correct the "~1 ms" estimate with the measured number**), add a short
completion handoff in `docs/`, commit in the local Mac repo
(`/Users/deven367/projects/agent-handoffs`, push), `git pull` on g37.

## Rollback

`cd ~/projects/tinygrad-src && git checkout 38342a3be -- tinygrad/llm/kernels/nv.py tinygrad/llm/kernels/nv_q4k.py tinygrad/llm/kernels/nv_q6k.py`

## Environment cheat sheet

- Work node: `ssh g37` (H100 SXM5 80GB, Slurm-gated — check `squeue -u demistry`; job
  `interact`/`h100-debu` has a 1 h limit, renew with `scontrol requeue <jobid>`).
  `node-quartz` = same host. g38 is the alternate H100 node.
- tinygrad: `~/projects/tinygrad-src`; Makefile targets from `~/projects/agent-handoffs`
  (identical to the Mac copy).
- **f32 residual stream** end-to-end (see prior handoff §1) — don't "fix" dtypes.
- Unrolling trap: custom-kernel Python loops unroll to straight-line UOps; keep per-thread
  register state bounded (the `ponytail:` comments in nv.py mark the ceilings).
- All 32 lanes must stay active for stores (no predication) — nv.py appendix / prior
  handoffs document the gater bug.

## After Lever A (remaining ranked list, re-profile first)

1. TODO 3: multi-warp grid-fused RMSNorm + Q8 quantize (~0.5–0.8 ms).
2. Fused 2nd residual add (`h + ffn_out`) into next block's `attn_norm` (cross-block
   restructure of `Transformer.forward`; ~64 more E_* launches).
3. kstat re-profile to re-rank (GEMV section was ~7.2 ms of 13.57).
