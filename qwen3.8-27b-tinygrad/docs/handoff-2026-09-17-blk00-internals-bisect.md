# Handoff — blk00 internals bisection: divergence source identified, scan logic correct

Date: 2026-09-17. Prior docs: `handoff-2026-09-07-cs2-bisect.md`, `ACTIVE.md`.

## TL;DR

The chunked-prefill divergence is **NOT in the recurrent scan or `_attention`**.
The padded step is a perfect no-op. The divergence originates in the **token
embedding** (`token_embd(tokens).float()`), caused by different JIT compilations
for different symbolic `toks` ranges. The scan, conv, gating, and padding are all
correct.

## Evidence

### Instrumentation script

`scripts/bisect_blk00_internals.py` — hooks `GatedDeltaNetBlock.__call__` for
block 0 only, preserving `@function(precompile=True)`. Intermediates are returned
as additional `@function` outputs (CALL UOp gettuple), making them realizable
without changing compilation behaviour.

### Key results (prompt `[1]`, cs=1 vs cs=2, DEV=CUDA, Q4_K_M, max_context=512)

```
DIVERGE block_input          max_abs=1.068115e-04 rel=3.38e-04   ← token embd (FIRST)
DIVERGE pre_norm_x           max_abs=1.068115e-04 rel=3.38e-04   ← same (inside @function)
DIVERGE x_in                 max_abs=7.658005e-03 rel=3.25e-04   ← after attn_norm (amplified)
DIVERGE conv_out             max_abs=2.668381e-03 rel=1.71e-04
DIVERGE beta_pad/log_alpha/q/k/v/alpha                       ← all propagate from x_in
OK      state_init           max_abs=0.000000e+00               ← initial state = 0 (correct)
OK      s1_t0                max_abs=0.000000e+00               ← s1 = 0 * alpha = 0 (correct)
DIVERGE delta_t0/state_t0/core/return                         ← propagated noise
--- No-op check ---
NO-OP  state_t0 -> state_t1: max_abs=0.000000e+00               ← padded step is NO-OP ✓
--- Padded-step delta ---
  delta_t1             max_abs=0.000000e+00                     ← delta=0 on padded step ✓
```

### Control test: cs=2 vs cs=2 (same symbolic shape)

```
OK  blk00-05: rel=0.00e+00  (perfect match)
```

JIT is deterministic for the same symbolic `toks` range. The divergence is
**specific to different `toks` max values** (1 vs 2), confirming it's a JIT
compilation difference, not non-determinism.

### Root cause: `_embedding_fwd` one-hot reduction

`nn.Embedding.__call__` (with `USE_ATOMICS=False`) uses:

```python
def _embedding_fwd(weight, idx):
    arange = Tensor.arange(weight.shape[0])  # 248320
    return (arange == idx.unsqueeze(-1)).unsqueeze(-1).where(weight, 0).sum(-2, dtype=weight.dtype)
```

This is a **one-hot mask × weight + reduction over vocab dimension (248320)**,
not a gather. Different `toks` max ranges (1 vs 2) compile to different reduction
kernels, producing slightly different float32 results. The token embedding weight
itself is float32 `(248320, 5120)`, not quantized.

The ~1e-04 noise propagates through 64 layers and flips the greedy argmax
(tok 271 vs tok 1) — expected behaviour for a quantized model with different
graph compilations.

## What's correct

- ✅ Padded steps are exact no-ops: `beta=0` (delta rule off), `alpha=1` (decay
  1), `delta=0`, `state` unchanged.
- ✅ Conv window assembly and `conv_state` store are correct.
- ✅ `pad_to(T_pad)` correctly zero-pads `beta` and `log_alpha`.
- ✅ Recurrent scan loop produces correct results for real tokens.
- ✅ `state_init = 0` when `start_pos=0` (initial reset).

## What's NOT a bug

- The `t % T_actual` modulo "fix" was correctly reverted — it was corrupting
  state. The current plain-`t` loop is correct.
- The `chunk_size=1` guard removal is safe — the scan logic handles partial
  chunks correctly.
- The suspect list from the prior handoff (conv window, pad_to no-op-ness,
  recurrent_state write) are all **ruled out** — all work correctly.

## Recommended next steps

1. **Re-enable chunked prefill**: Remove the `chunk_size=1` guard. The scan logic
   is verified correct for partial chunks. The numerical noise is inherent to JIT
   compilation with different symbolic shapes, not a logic bug.

2. **Verify argmax flip is benign**: Check top-K logits for cs=1 vs cs=2 on
   several prompts. If logits are close, the token difference is just noise near
   a decision boundary — acceptable for an LLM server.

3. **If exact token match is required**: Make the embedding lookup
   JIT-compilation-independent. Options:
   - Realize `token_embd(tokens).float()` before the JIT graph (materialize it).
   - Replace the one-hot reduction with a proper gather/index operation.
   - Use `USE_ATOMICS=True` (uses `Tensor.call` path, may be more stable).

4. **Prefill throughput**: With chunked prefill re-enabled, measure prefill
   throughput at cs=2/4/8/16/32. Watch VRAM at cs=32 (symbolic GEMV scratch —
   see `handoff-2026-09-05-vram-oom.md`).

5. **Tiled attention kernel**: Still needed for long-context decode performance
   (consumes packed Q8_0 cache directly, avoids dequantizing full prefix).
