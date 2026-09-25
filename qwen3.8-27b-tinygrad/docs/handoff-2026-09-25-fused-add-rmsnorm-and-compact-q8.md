# Handoff: Fused Residual Add+RMSNorm and Compact Q8 Activation Layout (2026-09-25)

**Hardware**: NVIDIA H100 SXM5 80GB (`g37.quartz.uits.iu.edu`, Slurm job `interact`, partition `h100-debu`)
**tinygrad**: `c14c50207` (pre-change) → **`38342a3be`** (this session), branch `qwen27b-nv-q8-kernel`
**Model**: `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (15.65 GiB, Q4_K_M)

Prior handoff: [`handoff-2026-09-25-actions-1-2-3-completed.md`](handoff-2026-09-25-actions-1-2-3-completed.md)

---

## 1. What landed (`38342a3be`, 2 files, +57/−6)

### TODO 1 — Fused residual add + RMSNorm (`nv_add_rmsnorm`)

`FFNBlock.__call__` (`tinygrad/llm/model.py`) was:

```python
h = x + self._attention(self.attn_norm(x), start_pos)          # E_* elementwise add
return (h + self._feed_forward(self.ffn_norm(h))).contiguous() # nv_rmsnorm re-reads h from DRAM
```

Now:

```python
attn_out = self._attention(self.attn_norm(x), start_pos)
h, h_normed = nv_add_rmsnorm(x, attn_out, self.ffn_norm.weight, self.ffn_norm.eps)
return (h + self._feed_forward(h_normed)).contiguous()
```

New `_add_rmsnorm_kernel` (`tinygrad/llm/kernels/nv.py`): one warp per token row; each lane
computes `v = x[idx] + res[idx]` once, accumulates `v²` for the norm, then stores both the
residual sum `h` and `rmsnorm(h) * weight`. The per-lane sums are single UOp values reused in
both passes, so the renderer keeps them in registers — no DRAM round-trip of `h` between the
add and the norm.

Per token row (dim 5120, f32): unfused = 2 kernels moving 4 loads + 2 stores (120 KB);
fused = 1 kernel moving 2 loads + 2 stores (80 KB). Eliminates 64 `E_*` add kernels per
decode step.

**Register pressure (the unrolling trap, revisited):** the residual stream is **f32**
(probe-verified: `Transformer.forward` starts `token_embd(tokens).float()` and nothing casts
back; only GatedDeltaNet internals go f16 and back). The 27B shape (dim 5120 → 160 elems)
holds 160 f32 sums per thread (~200 regs, under the 255 ceiling — no spill observed).
Marked with a `ponytail:` comment: a wider dim that spills should reload `x`/`res` in the
store pass (like `_rmsnorm_kernel`), costing 2 loads/elt.

**Bit-exactness:** the fused sum keeps the source dtype and operation order of the
standalone `E_*` add; the norm uses the identical formula and association as
`_rmsnorm_kernel` (`(total/dim + eps).rsqrt()`, `(v*norm)*w`). Outputs are bit-identical to
the unfused path (verified, below).

### TODO 2 — Compact Q8 activation layout

`_q8_quantize` allocated **32 uint32 words + 32 f32 scales per 32-element group** (256 B
written), but every GEMV consumer only reads words 0–7 and scale word 0 (36 B). Lanes 8–31
were storing duplicate/garbage words just to fill the 128 B coalesced lines.

Now: `q` is `(tokens, groups, 8)` u32 (32 B/group), `scale` is `(tokens, groups, 1)` f32.
Lanes 0–7 store the valid words; lanes 8–31 duplicate word 7 (same value — all 32 lanes
active, no warp predication, per the nv.py gater-bug lesson). Scale: all lanes store the
identical value (one 32 B sector). Write traffic per group: 256 B → 36 B.

Verified no consumer reads beyond word 8: `q8_0_linear` (`_load_lanes(xq[t,g,0], 8)`),
Q4_K v4/v2/scalar (word idx ≤ 7), Q6_K v2/scalar (w0/w1 ≤ 7), sweep scripts. `_load_lanes`
derives offsets from the live buffer shape, so it adapts automatically.

## 2. Verification

### Smoke (new kernels, direct, on g37)

- `nv_add_rmsnorm` **bit-exact** vs the unfused production path (`x + r` + `nv_rmsnorm`) for
  dim ∈ {896, 2048, 5120} × tokens ∈ {1, 3}: all `h` and `hn` bit-exact; ~1e-7 vs f64 math.
- Q8 compact layout **value-identical** to the old 32-word layout (old module loaded
  alongside, same inputs): q words and scale bit-exact for in_features ∈ {5120, 17408, 896};
  ~6e-8 rel vs numpy reference.

### Unit sweeps (`make test-units`)

`test_coop_q4k.py` 8/8 OK (rel ≤ 1.9e-6, f32 accumulation-order noise) and `sweep_q6k.py`
7/7 OK (maxabs ≤ 3.1e-5, scaled_rel ≤ 2.4e-7) — **ALL OK** with the compact layout.

### Full-model parity (`make parity`, cs=1 prefill, 23-token prompt)

```text
cs=1 argmax=271
cs=1 top5=[(271, 34.0718), (25, 19.5536), (11751, 18.7959), (248044, 17.0679), (198, 16.6857)]
cs=1 n=248320 max=+34.0718 min=-12.4540 sum=-800723.00
```

Identical to the golden values from `handoff-2026-09-25-actions-1-2-3-completed.md` —
**bit-exact across all 248,320 vocab logits**.

## 3. Benchmarks (clean: `make stop` first, `make serve` after)

| Engine (g37, H100 SXM5 80GB) | Decode tok/s | ms/tok | Parity |
|---|---:|---:|:---:|
| llama.cpp (`make bench-llama`, tg20) | **85.87 ± 1.00** | 11.65 | 100% (reference) |
| **tinygrad `38342a3be`** (`make bench-tg`) | **73.69** | **13.57** | **85.8%** |
| tinygrad `c14c50207` (previous session) | 68.79 | 14.54 | 79.8% |

- This session: **68.79 → 73.69 tok/s (+7.1%), −0.97 ms/tok** (14.54 → 13.57).
- Remaining gap to llama.cpp: **1.92 ms/tok** (was 2.94).
- Fast iteration (`make bench-tg-05b`, 0.5B): 109.02 tok/s (documented baseline 108.33 —
  within noise; measured with the server resident, 27B number is the clean one).

### Benchmark hygiene (IMPORTANT, new finding)

The user's `llama-server` (Q8_K_XL + MTP, 44.7 GiB, port 9932) had been resident on g37 all
day and was bursting to **44% SM utilization** — it cut `llama-bench` from 85.87 to
**48.51 tok/s** (−43%) while barely touching tinygrad (73.69 clean vs 73.76 resident).
**Always `make stop` before any benchmark and `make serve` after.** (Server was restored:
health OK, same cmdline/args, 44.4 GiB.)

## 4. Environment deltas vs previous handoffs

- Work node is **g37** via `ssh g37` (Slurm-gated: needs an active job; partition
  `h100-debu`, 1 h limit — `scontrol requeue` to renew). `node-quartz` alias = same node.
  Previous docs say g38; both are H100 SXM5 nodes on Quartz.
- Repros: tinygrad at `~/projects/tinygrad-src`, agent-handoffs at `~/projects/agent-handoffs`
  (Makefile byte-identical to the local Mac copy — verified by diff).
- `~/bin/git-personal` sets repo-local identity but does **not** forward args — run it
  alone, then plain `git add`/`git commit`.
- **Residual stream is f32 end-to-end** (see §1 TODO 1). Any future "store activations in
  f16 to halve traffic" idea must account for this: the E_* adds, residual stores, and
  norms all run f32.

## 5. Remaining gap & next steps

Expected kernel census: ~2,060 → ~1,996 per step (−64 E_* adds; q8 call count unchanged,
stores 7× smaller).

Ranked levers (re-profile with the kstat.py pattern on a DEBUG=2 log before starting):

1. **Lever A (structural, largest remaining): compact GEMV output buffers.** Every GEMV
   (`_decode_linear` family: q8_0/Q4_K/Q6_K) stores `(rows, 32)` f32 per output row —
   128 B written, only word 0 consumed by `result[..., 0]` (~30–40 MB/tok of pure write
   waste across 572 GEMV calls). Restructure the store or hand the consumer a compact
   output. Estimate up to ~1 ms/tok; touches all three GEMV families.
2. **TODO 3 (carried, ~0.5–0.8 ms/tok): multi-warp grid-fused RMSNorm + Q8 quantize** —
   160-warp grid instead of a single-warp loop (avoids UOp-unrolling register spilling).
3. **Fused 2nd residual add (`h + ffn_out`) into the next block's `attn_norm`** —
   cross-block restructure of `Transformer.forward` (yield `(residual, normed)` pairs;
   `output_norm` consumes the final residual). Removes ~64 more E_* launches.
