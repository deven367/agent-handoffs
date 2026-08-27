# Handoff — Qwen3.8-27B tinygrad / Unsloth Q4 optimization (2026-08-27)

## Executive state

Work is **not complete**. This session stabilized the remote environment, repaired the home Makefile target, implemented and verified exact Q3_K/IQ4_NL GGUF loaders, and implemented NVIDIA Q5_K/IQ4_XS GEMV kernels that pass synthetic/reference sweeps. A full-model logit A/B exposed a blocker: **the Unsloth model produces all-NaN logits whenever any existing NVIDIA custom-quant path is enabled**, while the all-generic path is finite. The root cause is not yet isolated. Do not benchmark or serve the Unsloth model until this is fixed.

No GPU process is intentionally left running. Current assigned GPU: one NVIDIA L40S.

## Remote repository

- Host: `node-lair`
- Worktree: `/u/demistry/tinygrad-src`
- Branch: `qwen27b-nv-q8-kernel`
- Tracking: `fork/qwen27b-nv-q8-kernel`, branch is **ahead by 6 commits**
- Remote worktree at handoff:

```text
 M tinygrad/llm/gguf.py
 M tinygrad/llm/kernels/amd.py
?? tinygrad/llm/kernels/nv_iq4xs.py
?? tinygrad/llm/kernels/nv_q5k.py
```

Tracked diff summary:

```text
tinygrad/llm/gguf.py        | 34 ++++++++++++++++++++++++++++------
tinygrad/llm/kernels/amd.py | 15 +++++++++++++--
```

Recent commits:

```text
29a306ec6 P2 chunked-prefill gate removal (UNVERIFIED)
f99dab110 Q3_K placeholder loader (BROKEN in commit; corrected by current uncommitted diff)
1e78b0d54 Q3/IQ4 claim commit (MISLEADING; corrected by current uncommitted diff)
65e09942f revert false Q6_K loader fix
1fe4ba369 verified NVIDIA Q6_K GEMV + routing
2d46ea489 verified NVIDIA Q4_K routing
ad94c9619 verified NVIDIA Q4_K kernel
6b784772d Q8_0 streaming-load optimization
c801cc659 Q8_0 routing
55c9213c2 BEAM_CACHE implementation
```

**Do not discard the current working tree.** `f99dab110` by itself returns zeros for Q3_K; the uncommitted `gguf.py` diff replaces that placeholder with the exact decoder.

## Model inventory

```text
/scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf          16,464,440,224 bytes
/scratch/local/demistry/models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf  16,810,714,496 bytes
/scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf         31,457,991,680 bytes
/scratch/local/demistry/models/gemma-3-270m-it-UD-Q8_K_XL.gguf        471,104,544 bytes
```

Correct Unsloth UD-Q4 tensor census (866 tensors):

```text
F32       360
Q5_K      131
IQ4_XS    117
Q8_0      106
Q4_K      104
Q6_K       30
Q3_K        7
IQ4_NL      7
IQ3_S       4
```

`/u/demistry/p1/inspect_gguf.py` has an obsolete enum-name table after type 14. Its raw counts are useful, but translate types with current llama.cpp `ggml.h`: 20=IQ4_NL, 21=IQ3_S, 22=IQ2_S, 23=IQ4_XS.

## Completed this session

### 1. Environment cleanup

Two stale interactive `llama-cli` jobs were stopped. A canceled isolation process was also stopped. GPU compute-app list was empty at handoff.

### 2. Home Makefile repaired

File: `/u/demistry/Makefile` (outside git)

Targets now parse correctly:

```bash
make -f ~/Makefile serve-tg
make -f ~/Makefile status-tg
make -f ~/Makefile logs-tg
make -f ~/Makefile stop-tg
```

`make -n -f ~/Makefile serve-tg` passes. Defaults:

```text
TG_MODEL=/scratch/local/demistry/models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf
TG_PORT=8888
TG_CWD=/u/demistry/tinygrad-src
```

It does **not** provide MTP. Do not point it at the Unsloth model until the NaN blocker is fixed.

### 3. Exact Q3_K and IQ4_NL loaders

Modified: `tinygrad/llm/gguf.py`

- Added `_GGML_QUANT[11] = (256, 110)`.
- Added exact Q3_K scale unpack, hmask/low-bit reconstruction, and f16 scaling.
- Added `_GGML_QUANT[20] = (32, 18)`.
- Added exact IQ4_NL nonlinear-LUT decode using `kvalues_iq4nl`.
- Updated supported-type documentation.

Verification script:

```text
/u/demistry/p1/check_unsloth_loaders.py
```

Run:

```bash
cd /u/demistry/tinygrad-src
python3 /u/demistry/p1/check_unsloth_loaders.py
```

Observed:

```text
random Q3_K:    maxabs=0 rel=0 exact=True
random IQ4_NL:  maxabs=0 rel=0 exact=True
real Q3_K:      maxabs=0 rel=0 exact=True
real IQ4_NL:    maxabs=0 rel=0 exact=True
ALL LOADER CHECKS OK
```

Real samples came from `blk.0.ffn_up.weight` (Q3_K) and `blk.1.ffn_down.weight` (IQ4_NL).

Model construction smoke now succeeds:

```bash
cd /u/demistry/tinygrad-src
python3 -c 'from tinygrad.llm.model import Transformer; m,kv=Transformer.from_gguf("/scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf",128); print(len(m.blk),kv.get("qwen35.nextn_predict_layers"))'
```

Observed: `64 1`.

### 4. NVIDIA Q5_K and IQ4_XS GEMV kernels

New files:

```text
tinygrad/llm/kernels/nv_q5k.py
tinygrad/llm/kernels/nv_iq4xs.py
```

Modified routing: `tinygrad/llm/kernels/amd.py`

- NVIDIA now claims Q5_K (176 bytes/256 values) and routes to `q5_k_linear`.
- NVIDIA now claims IQ4_XS (136 bytes/256 values) and routes to `iq4_xs_linear`.
- Removed the erroneous Q3_K/IQ4_NL entries from AMD `QUANT_SIZES`; there are no custom kernels for those types yet.
- Q5_K uses packed qh/qs plus the existing K-scale/min scheme and signed `dp4a` activation dots.
- IQ4_XS uses an exact 16-entry nonlinear LUT passed lazily into the custom kernel. Do not add `.realize()` in `_iq4_lut`; `@function` scheduling disables device allocation.

Sweep:

```text
/u/demistry/p1/sweep_nv_unsloth.py
```

Run:

```bash
cd /u/demistry/tinygrad-src
python3 /u/demistry/p1/sweep_nv_unsloth.py
```

Observed all 10 cases pass:

```text
Q5_K:   256x256, 512x1024, 768x768, 1024x5120, 6144x5120 — all rel <= 1.25e-7
IQ4_XS: same shapes — all rel <= 8.7e-8
ALL OK
```

The initial CUDA `__byte_perm` IQ4 implementation was incorrect because AMD/CUDA permutation semantics differ. It was replaced with a small lazy uint8 LUT buffer; current sweep is exact within accumulation order.

## Current blocker: custom full-model logits are NaN

Harness:

```text
/u/demistry/p1/unsloth_logits.py
```

Generic result:

```text
generic: finite=True argmax=5328 min=-14.859167 max=10.191113 wall=45.248s
```

Custom configurations all returned all-NaN logits:

```text
custom:   all NaN
q5only:   all NaN
iq4only:  all NaN
oldonly:  all NaN
```

Mode meanings:

- `custom`: Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS custom paths.
- `q5only`: name is imperfect; it disables new IQ4_XS but still leaves existing Q8/Q4/Q6 plus Q5 custom.
- `iq4only`: disables new Q5 but leaves existing Q8/Q4/Q6 plus IQ4 custom.
- `oldonly`: disables both new kernels, leaving only existing Q8/Q4/Q6 custom.

Because `oldonly` is also NaN, the new kernels are **not proven to be the cause**. At least one pre-existing Q8/Q4/Q6 custom path behaves incorrectly on a tensor shape/layout in the Unsloth model, or packed-buffer engagement is wrong for this file.

A slow per-block harness exists:

```text
/u/demistry/p1/unsloth_isolate.py
```

`q8only` was running when the handoff was requested and was canceled/cleaned. It had not produced a result.

### Recommended immediate debugging path

Avoid another full 64-block generic isolation first. Build real-tensor micro A/B checks:

1. Load the Unsloth model once and locate the first custom-eligible layers in block 0.
2. Compare each individual projection against `ggml_data_to_tensor(raw) @ x` with a deterministic input.
3. Start with Q8_0 `ssm_alpha`/`ssm_beta`, then Q4_K/Q6_K.
4. Realize/check finiteness after every projection, not only after each transformer block.
5. Verify `Linear.set_quantized` selected the expected `ggml_type`, exact raw byte count, buffer offset, and typed view length.
6. Once existing Q8/Q4/Q6 are finite, enable Q5_K, then IQ4_XS independently.

Do not commit or report the Unsloth kernels as model-verified until full logits are finite and match generic logits within a documented tolerance.

## Known verified older work

These existed before this session and should be preserved:

- NVIDIA Q8_0 GEMV: prior exact sweeps/token A/B; about 20.9 tok/s on the prior Q8 model.
- NVIDIA Q4_K GEMV: prior sweep 8/8 and token A/B 32/32.
- NVIDIA Q6_K GEMV (`1fe4ba369`): prior sweep 7/7; Q6 path around 7.3 ms/step on the prior Q4_K_M model.
- The claimed Q6 loader bug was false; `65e09942f` correctly reverts that bad fix.

The prior 28.5 tok/s result is for the Uncensored/OBLITERATED-style Q4_K_M mix, **not** a verified Unsloth result.

## Still pending after the NaN fix

### Chunked prefill (`29a306ec6`)

The commit removes the recurrent-model safety gate globally. It has not been correctness- or performance-validated on NV. The generic recurrent path unrolls the scan for a chunk; validate token/logit equivalence and compile/runtime cost. Restore the gate if it regresses until an NV fused scan exists.

### NVIDIA DeltaNet

No NV DeltaNet kernel exists. Ignore prior `/tmp/nv_ssm.py` “framework” claims; it was comments only and is not in the repo. The correct routing point is the recurrent branch around `model.py:326`, not attention/flash-attention around line 187.

Needed:

- actual fused recurrent scan/decode UOp kernel;
- recurrent-state and output equivalence tests;
- decode/prefill benchmark.

### BEAM_CACHE

Implementation exists at commit `55c9213c2`, but no validated persistent training/replay for this 27B model exists. Train with a persistent disk-cache location, prove `beam cache hit` on replay, verify identical outputs, then benchmark without search cost.

### MTP

Not implemented. Current loader/model intentionally subtracts `nextn_predict_layers` at `model.py:435`. The Unsloth file has one nextn layer (`blk.64.nextn.*`). Required work:

- retain and map the nextn module weights;
- implement draft logits;
- speculative acceptance/resampling;
- attention KV, convolution state, recurrent state, and token-cache rollback on rejection;
- integrate into `generate()` and API serving;
- compare deterministic outputs and throughput with llama.cpp `--draft-mtp`.

### Same-file comparison

No valid tinygrad-vs-llama.cpp benchmark exists for the Unsloth Q4 file. After model equivalence:

- use `llama-bench` or a noninteractive `llama-cli` command (previous interactive commands hung);
- benchmark same GPU, same file, same prompt/context;
- run llama.cpp without and with MTP;
- run tinygrad without and with MTP;
- report prefill and decode separately.

## Scripts and artifacts

```text
/u/demistry/p1/check_unsloth_loaders.py   exact loader random + real-block tests
/u/demistry/p1/sweep_nv_unsloth.py       Q5_K/IQ4_XS GEMV sweep
/u/demistry/p1/unsloth_logits.py          full-model custom/generic logit harness
/u/demistry/p1/unsloth_isolate.py         slow per-block isolation harness
/u/demistry/p1/logits_generic.npy         finite generic reference logits
/u/demistry/p1/logits_custom.npy          NaN custom logits
/u/demistry/p1/logits_q5only.npy          NaN isolation output
/u/demistry/p1/logits_iq4only.npy         NaN isolation output
/u/demistry/p1/logits_oldonly.npy         NaN existing-kernel isolation output
```

## Suggested first commands for the next agent

```bash
ssh node-lair
cd /u/demistry/tinygrad-src
cat AGENTS.md
git status --short --branch
git diff -- tinygrad/llm/gguf.py tinygrad/llm/kernels/amd.py
python3 /u/demistry/p1/check_unsloth_loaders.py
python3 /u/demistry/p1/sweep_nv_unsloth.py
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Then implement a real-tensor per-linear comparison to isolate the first bad existing custom kernel. Do not begin MTP or DeltaNet until generic-vs-custom logits for the base Unsloth model are finite and equivalent.

## Delivery state

- Remote tinygrad changes from this session are **uncommitted**.
- The home Makefile repair is outside git.
- The local `agent-handoffs` repo already had many uncommitted/untracked files before this handoff. This handoff file itself is new.
- No push was performed this session.
