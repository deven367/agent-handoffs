# Handoff — 2026-09-06: Chunked prefill codegen fix + remaining issues

## What was done this session

### 1. Fixed: NVRTC `expected a "}"` compile error (commit `39a790966`)

**Root cause:** Loops with symbolic bounds (`start_pos + toks`) have their `Ops.END` uops dropped from the `linearize` toposort when the END is not a transitive dependency of the SINK. The `_render` function in `cstyle.py` processes the RANGE (emits `for (...) {`) but never the matching END (emits `}`), producing malformed CUDA with unbalanced braces.

**Fix:** `tinygrad/renderer/cstyle.py`, `_render` method — after the uop loop, emit closing braces for any remaining `depth > 1`:

```python
# ponytail: missing END uops for symbolic-bounded loops leave depth > 1
while depth > 1:
  depth -= 1
  kernel.append("  "*depth + "}")
```

**Verified:** Q4_K_M, FP16 KV, CUDA backend, `chunk_size=2` now compiles via NVRTC. `chunk_size=1` still produces token 39840 (unchanged). Committed on `qwen27b-nv-q8-kernel` branch as `39a790966`.

### 2. Found + partially fixed: `cuGraphAddKernelNode` invalid argument

After the brace fix, `chunk_size=2` compiles but fails at `cuGraphAddKernelNode` with `CUDA Error 1, invalid argument`.

**Root cause:** `CUDAGraph.__init__` (line 20 of `tinygrad/runtime/graph/cuda.py`) computes launch dimensions with `{v: 0 for v in self.vars}` — substituting 0 for all symbolic variables. When a kernel has a symbolic dimension derived from `toks` (e.g. `E_toks_5_2_4_2_16_2_2`), substituting `toks=0` produces `global_size=(2, 5, 0)` — a zero grid dimension, which CUDA rejects.

**Debug output confirmed:**
```
FAILED KERNEL NODE: E_toks_5_2_4_2_16_2_2 res=1
  global_size=(2, 5, 0) local_size=(4, 2, 16) smem=0
  vars=[('toks', 0)]
```

**Partial fix (NOT committed — reverted):** Changing line 20 to use `{v: 1 for v in self.vars}` and clamping dimensions to `max(1, x)` avoids the CUDA graph error. This was tested and allows `chunk_size=2` to run end-to-end.

**Why not committed:** The fix produces **incorrect output**. `chunk_size=2` generates token 220 while `chunk_size=1` generates token 39840 for `list(range(64))`. The mismatch is a correctness bug in the chunked prefill path, not in the graph fix itself. The graph fix is necessary but not sufficient.

### 3. Investigated: chunked prefill correctness

Tested with small prompts to isolate the mismatch:

| Prompt | chunk_size=1 | chunk_size=2 | Match? |
|--------|-------------|-------------|--------|
| `[1, 2]` | 220 | 220 | ✅ |
| `[1]` | 271 | 1 | ❌ |
| `list(range(64))` | 39840 | 220 | ❌ |

When `chunk_size=1` with prompt `[1]`: the model processes 1 token, start_pos=0, T=1, and the `while` loop in `generate` immediately appends the output token. The model enters the `rollout_jit` path (T=1).

When `chunk_size=2` with prompt `[1]`: the model processes 1 token at `start_pos=0` with `n_toks=min(2, 1-0)=1`, so T=1. BUT `v_toks` was created with `UOp.variable("toks", 1, chunk_size)` = `UOp.variable("toks", 1, 2)`. The JIT captures the prefill_jit with `toks` bound to 1, but the symbolic max is 2. The `T_pad` in `_attention` uses `x.max_shape[1]` which is the **max** of the symbolic range (2), not the bound value (1). So the model pads to 2 tokens and processes a phantom second token.

**Root cause of correctness bug:** The `GatedDeltaNetBlock._attention` method at line 318:
```python
T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size
```
When `chunk_size=2` and only 1 token is available, the tensor is sliced to 1 token, but `T_pad` = 2 (the symbolic max). The recurrent scan at lines 365-369 unrolls `for t in range(T_pad)` = `range(2)`, processing a zero-padded second step. While zero-padded steps are designed to be no-ops (beta=0, log_alpha=0), the **conv_state** handling may not correctly skip the padded step, and the recurrent state update is incorrect.

This is the same class of bug that the `chunk_size=1` guard was protecting against. The guard exists because the non-AMD recurrent path unrolls in Python and the symbolic padding semantics don't match the sequential execution semantics.

## Current state of tinygrad-src

- Branch: `qwen27b-nv-q8-kernel` on `node-lair:/u/demistry/tinygrad-src`
- HEAD: `39a790966 fix(renderer): emit missing closing braces for dropped END uops`
- Working tree: **clean** (all debug patches reverted)
- The `chunk_size=1` guard at `model.py:534` is **still in place**

## What the next agent must do

### Priority 1: Fix chunked prefill correctness for recurrent models

The core issue: when `chunk_size > 1` but fewer tokens than `chunk_size` remain in the prompt, the symbolic `T_pad` pads to `chunk_size` and the recurrent scan processes phantom tokens. The padded steps are supposed to be no-ops, but the recurrent state is corrupted.

**Approach A (simplest):** Ensure `generate()` always passes exactly `chunk_size` tokens to the model, padding the input with zeros if needed, and only extract the output from the last real token. This requires changing the generate loop to pad the prompt to a multiple of `chunk_size`.

**Approach B (targeted):** Fix `GatedDeltaNetBlock._attention` so that `T_pad` uses the bound value of `toks`, not the max. This means `T_pad = resolve(x.shape[1])` when possible, falling back to `x.max_shape[1]` for symbolic shapes.

**Approach C (correct):** Port the AMD `gated_delta_prefill` fused kernel to NVIDIA, which handles arbitrary token counts in a single kernel and avoids the Python unrolled scan entirely. This is plan 09 Phase 4.

### Priority 2: Fix the CUDA graph zero-dimension issue

Even after the correctness fix, the CUDA graph still needs the launch-dims fix. The proper fix is in `tinygrad/runtime/graph/cuda.py` line 20:

```python
# Current (broken for symbolic dims):
global_size, local_size = ast.arg.launch_dims({v: 0 for v in self.vars})

# Fix: use 1 as placeholder, then clamp to max(1, x)
global_size, local_size = ast.arg.launch_dims({v: 1 for v in self.vars})
global_size = tuple(max(1, x) for x in global_size)
if local_size: local_size = tuple(max(1, x) for x in local_size)
```

The real dimensions are updated at runtime via `updated_launch_dims` (line 60-62), so the initial values just need to be valid for graph construction. This fix is safe — it only affects the initial graph node creation, not runtime execution.

### Priority 3: Remove the `chunk_size=1` guard

Once priorities 1 and 2 are fixed and verified:
1. Remove line 534 of `model.py`: `if self.has_recurrent_block and not amd_custom_kernels_supported(...): chunk_size = 1`
2. Run A/B: same prompt, `chunk_size=1` vs `chunk_size=2`/`4`/`8`/`16`/`32`, verify identical greedy tokens
3. Measure prefill throughput improvement
4. Watch for VRAM OOM at `chunk_size=32` (the symbolic GEMV scratch issue from the VRAM-oom handoff)

## Key files

```text
tinygrad/renderer/cstyle.py        — brace fix (committed 39a790966)
tinygrad/runtime/graph/cuda.py     — needs launch_dims fix (line 20)
tinygrad/llm/model.py              — guard at line 534, generate loop 533-552
tinygrad/llm/model.py:318          — T_pad = x.max_shape[1] (correctness bug)
tinygrad/llm/model.py:355-373      — recurrent scan unroll (Python loop)
tinygrad/llm/kernels/amd.py:482    — flash_attention assert T_pad % 32 == 0
```

## Reproduction commands

```bash
# On node-lair, tinygrad at /u/demistry/tinygrad-src
# chunk_size=1 (works, token 39840):
python3 -c "
import tinygrad.llm.model as m
model, _ = m.Transformer.from_gguf('/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf', max_context=512, cache_type='f16')
print(next(model.generate(list(range(64)), chunk_size=1)))
"

# chunk_size=2 (compiles but wrong output, token 220):
# Must remove guard at model.py:534 first
python3 -c "
import tinygrad.llm.model as m
m.amd_custom_kernels_supported = lambda _: True  # bypass guard
model, _ = m.Transformer.from_gguf('/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf', max_context=512, cache_type='f16')
print(next(model.generate(list(range(64)), chunk_size=2)))
"
```

## Notes

- The `{v: 0 for v in self.vars}` bug in cuda.py line 20 likely affects ALL backends that use CUDA graphs with symbolic dimensions, not just this model. The fix is broadly applicable.
- The `flash_attention` assert at `amd.py:482` (`T_pad % BLOCK_M == 0`) means `chunk_size` must be a multiple of 32 for attention blocks. This is fine for prefill but means arbitrary chunk sizes won't work without code changes.
- The brace fix in cstyle.py is a safety net, not a root-cause fix. The real fix would be ensuring `Ops.END` uops are always included in the `linearize` toposort. The safety net is sufficient for now.
