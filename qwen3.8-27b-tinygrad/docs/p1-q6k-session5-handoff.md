# Session 5 Handoff — Q6_K NV kernel verified; model choice paused (2026-08-27)

**Status: PAUSED on user request.** The Q6_K NVIDIA kernel and routing are implemented and
verified, but **not committed**. The llama.cpp token comparison was interrupted when the user
switched from the OBLITERATED quant to the Unsloth UD Q4_K_M quant, which this tinygrad cannot
currently load. The user asked to stop while deciding what to do next.

No managed GPU process remains active.

## 1. Remote worktree state

Repository: `/u/demistry/tinygrad-src`

- Branch: `qwen27b-nv-q8-kernel`
- HEAD: `2d46ea4895fc`
- Modified: `tinygrad/llm/kernels/amd.py` (`11 insertions, 2 deletions`)
- Untracked: `tinygrad/llm/kernels/nv_q6k.py`
- `tinygrad/llm/gguf.py` is **untouched**.
- Nothing from this session is committed.

`amd.py` changes:

1. Import `q6_k_linear`.
2. On NVIDIA, recognize packed Q6_K buffers by `decoded.numel() // 256 * 210`.
3. Store NVIDIA Q6_K packed weights as a `uint16` buffer view; AMD continues using `uint8`.
4. Route concrete and symbolic Q6_K inputs through `q6_k_linear`, including the existing
   `pad_to(...)/shrink(...)` symbolic-token pattern.

`nv_q6k.py`:

- 256 weights per 210-byte block, viewed as 105 `uint16` words.
- Uses `_q8_quantize`, `_decode_linear`, `_load_lanes`, `_nv_dp4a`, and `_nv_ldcs16` from
  `tinygrad.llm.kernels.nv`.
- Performs two dot/sum chains per 32-weight subgroup for the two int8 scales.
- Uses streaming 16-bit weight loads because the 210-byte block stride is 2-byte aligned but
  not generally 4-byte aligned.

**Important correction made before deployment:** the original draft computed a per-row/per-block
`base` but did not add it to any QL/QH/scale/d load. That would have read block zero for every
row and block. The deployed file adds `base` to every packed-weight access.

## 2. The claimed Q6_K loader bug was false

Do **not** apply the one-line loader fix proposed in `p1-q6k-handoff.md`.

The earlier session looked only at the final expression:

```python
xl.bitwise_or(xh)
```

but `xh` is already shifted at its definition in the current source:

```python
xl, xh = ..., q_to_uint8(..., 2).lshift(4)
```

Therefore the current operation is already `xl | (raw_xh << 4)`. `git blame` attributes that
shift to commit `67ed4c4eb3`; it was not added during this work.

The original `check_q6k_loader.py` also had independent harness defects:

1. GGUF metadata value types were parsed as one byte instead of `uint32`, and it attempted an
   extra string read per metadata entry.
2. Tensor offsets were treated as absolute instead of relative to the aligned GGUF data section.
3. Its llama.cpp Q3/Q6-style half-loop port did not advance QL/QH/scale bases for weights
   128–255.
4. Its final ndarray equality assertion was not a scalar assertion.

The corrected checker is local at `scripts/check_q6k_loader.py` and deployed at
`/u/demistry/p1/check_q6k_loader.py`.

### Real-byte result

On 64 real blocks from each of three Q6_K tensors in the copied OBLITERATED model:

- `output.weight`
- `blk.0.attn_qkv.weight`
- `blk.0.ffn_down.weight`

Results:

```text
Q6_K tensors: 67
tinygrad actual:          maxabs=0 maxrel=0 exact=True
source formula (xh << 4): maxabs=0 maxrel=0 exact=True
hypothetical unshifted:   exact=False
Q6_K LOADER CHECK: current tinygrad == ggml port (exact)
```

The hypothetical unshifted implementation differed by up to `0.10849` on the sampled model
bytes. Applying another shift in `gguf.py` would corrupt the loader.

## 3. The “vanishing model” incident was a filename typo

The previous handoff used this nonexistent source path:

```text
/scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf
```

The actual filename contains a hyphen, not a dot:

```text
/scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
```

Once the correct path was used, it was immediately readable. A stable copy was made and
size-verified here:

```text
/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
size: 16,810,705,952 bytes
```

There is no evidence from this session that the correctly named source file was being repeatedly
unlinked.

## 4. Kernel verification completed

### Full shape sweep

Harnesses:

- Local: `scripts/sweep_q6k.py`
- Remote: `/u/demistry/p1/sweep_q6k.py`

The harness packs random Q6 codes in `[-32, 31]`, random int8 scales, and an f16 block scale.
It independently checks the packer and a scalar llama.cpp dequant port, then compares the custom
kernel against an f32 reference matmul. Large output matrices use eight distinct row patterns to
keep host reference construction tractable while still testing row/block addressing.

All prescribed shapes passed:

```text
OK   256x256 t2       maxabs=7.62939e-06 scaled_rel=1.01172e-07
OK   512x1024 t4      maxabs=7.62939e-06 scaled_rel=1.05066e-07
OK   768x768 t1       maxabs=9.53674e-06 scaled_rel=2.54521e-07
OK   1024x5120 t1     maxabs=7.62939e-06 scaled_rel=6.38436e-08
OK   10240x5120 t1    maxabs=3.05176e-05 scaled_rel=1.06779e-07
OK   17408x5120 t1    maxabs=1.52588e-05 scaled_rel=5.97292e-08
OK   248320x5120 t1   maxabs=7.62939e-06 scaled_rel=9.67525e-08
ALL OK
```

This covers multi-token execution, a 24-group partial warp, all production Q6_K matrix shapes,
and the 1.04 GB lm_head packed matrix.

### Small-model token A/B

Harnesses:

- Local: `scripts/proxy_ab_q6k.py`
- Remote: `/u/demistry/p1/proxy_ab_q6k.py`

Model: `qwen3.5:4b`, prompt `The capital of France is`, greedy generation.

```text
CUSTOM:  q4_k_linears=131 q6_k_linears=22
GENERIC: q4_k_linears=0   q6_k_linears=0
MATCH 32/32 token positions
```

The prior expectation of “70 Q6_K linears” was wrong for this post-forward object count. The
model engaged 22 Q6_K `amd.Linear` objects. The earlier “70 unclaimed” diagnostic was not an
interchangeable count of engaged Q6_K objects.

## 5. OBLITERATED 27B performance results

Command family:

```bash
python3 -m tinygrad.llm \
  --model /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf \
  --max_context 512 --benchmark 20
```

Steady wall time after the two cold/capture iterations:

```text
34.83–34.94 ms/step
28.62–28.71 tok/s
~461 GB/s reported
```

Previous baseline from `p1-handoff.md`: `12.49 tok/s`, `80.08 ms/step`.
The end-to-end benchmark loop therefore improved by about **2.30×**.

### JIT graph timing

Detailed log: `/tmp/bench_q6k_jit_debug.log` (temporary; copy it out if the node may restart).

A steady iteration was split into these batches:

```text
batched 32      0.048 ms
batched 64      0.122 ms
batched 128     1.794 ms
batched 256     3.554 ms
batched 512     6.749 ms
batched 1024   13.980 ms
batched 397     7.937 ms   # final decode graph
cumulative     34.29 ms
```

The final `batched 397` line is the decode graph. It fell from the prior **20.3 ms** to about
**7.94 ms**, a **2.56× decode-graph speedup**. The first six graph batches also became faster
because Q6_K appears in those paths; their cumulative time is about 26.35 ms.

### Fresh no-JIT profile

Log: `/u/demistry/logs/p1_q6k_nojit.log`

Analyzer:

```bash
python3 /u/demistry/p1/kstat2.py /u/demistry/logs/p1_q6k_nojit.log 3
```

Average of the final three steps:

```text
GPU-time sum/step: 36.26 ms
kernels/step: 2413

nv_linear_q4_k       432/step   19.948 ms   55.0%
nv_linear_q6_k        65/step    7.298 ms   20.1%
generic r_*          967/step    7.172 ms   19.8%
generic E_*          452/step    1.144 ms    3.2%
nv_q8_quantize       497/step    0.699 ms    1.9%
```

The old generic Q6_K matmul families are gone. The new custom Q6_K total, `7.30 ms/step`, matches
the prior memory-bound estimate of roughly 7.5 ms. The profile observed 65 Q6_K calls rather than
the static-file count of 67; do not treat tensor count and one profiled decode-step count as
identical.

## 6. User-selected Unsloth Q4_K_M model is materially different

The user changed the requested llama.cpp cross-check model to:

```text
/scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf
size: 16,464,440,224 bytes
```

It has the same 866 tensors as the OBLITERATED file and is only 346,265,728 bytes smaller
(about 2.1%), but its quantization mix is very different.

### Tensor-type comparison

OBLITERATED:

```text
F32   360
Q4_K  439
Q6_K   67
```

Unsloth UD:

```text
F32       360
Q8_0      106
Q3_K        7   # unsupported by current tinygrad
Q4_K      104
Q5_K      131
Q6_K       30
IQ4_NL      7   # GGML type 20; unsupported by current tinygrad
IQ3_S       4
IQ4_XS    117
```

Tinygrad fails immediately while constructing the Unsloth model:

```text
ValueError: GGML type '11' is not supported!
```

Adding only Q3_K support is insufficient: seven IQ4_NL/type-20 tensors would fail next. No Q3_K
or IQ4_NL code was applied before the user paused the work.

Reference facts already located, if the user chooses to add support:

- Q3_K: 256 elements, 110 bytes: `hmask[32]`, `qs[64]`, `scales[12]`, `d:f16[2]`.
  llama.cpp reference: `ggml/src/ggml-quants.c::dequantize_row_q3_K`.
- IQ4_NL: 32 elements, 18 bytes: `d:f16[2]`, `qs[16]`; dequant uses the existing
  `kvalues_iq4nl` nonlinear LUT.

Treat these as a separate loader feature with independent reference tests and a separate commit,
not as an incidental addition to the Q6_K kernel commit.

## 7. llama.cpp cross-check status

Build:

```text
/u/demistry/llama.cpp/build/bin
```

The OBLITERATED model successfully loaded in `llama-server` on CUDA1 and listened on port 9942,
but no completion request was sent before the user switched models and requested a stop.

Device-selection gotcha: after passing `--device CUDA1`, llama.cpp sees a filtered one-device
list, so `--main-gpu` must be `0`, not `1`:

```bash
llama-server \
  --model MODEL \
  --ctx-size 512 --n-gpu-layers all \
  --device CUDA1 --split-mode none --main-gpu 0 \
  --parallel 1 --host 127.0.0.1 --port 9942 --no-warmup
```

The first attempt used `--main-gpu 1` and failed with:

```text
invalid value for main_gpu: 1 (available devices: 1)
```

The user reported both GPUs free at the pause point. No process started by this session remains
active.

### Tinygrad token artifact from OBLITERATED

Harnesses:

- Local: `scripts/generate_q6k_tokens.py`
- Remote: `/u/demistry/p1/generate_q6k_tokens.py`

Prompt: `The capital of France is`

Actual prompt token IDs:

```text
[760, 6511, 314, 9338, 369]
```

Tinygrad generated 16 greedy tokens:

```text
[11751, 13, 198, 57590, 369, 264, 3177, 303, 9338, 13, 198, 52971, 11, 279, 6511, 314]
```

Decoded:

```text
 Paris.
Paris is a city in France.
Therefore, the capital of
```

The first harness run passed the prompt list directly to `model.generate`, which mutates that list
by appending generated IDs. The deployed/local harness is now corrected to pass
`prompt_tokens.copy()`; the model was not rerun after that mechanical harness fix.

For a llama-server comparison, use the numeric prompt tokens to avoid BOS/tokenizer-policy
ambiguity and request raw generated IDs:

```json
{
  "prompt": [760, 6511, 314, 9338, 369],
  "n_predict": 16,
  "temperature": 0,
  "samplers": ["temperature"],
  "seed": 1234,
  "cache_prompt": false,
  "return_tokens": true
}
```

POST it to `/completion`, then compare the response `tokens` array exactly. This comparison has
**not** been run.

## 8. Artifacts

Local, in this handoff repository:

```text
kernels/nv_q6k.py
scripts/check_q6k_loader.py
scripts/sweep_q6k.py
scripts/proxy_ab_q6k.py
scripts/generate_q6k_tokens.py
```

Remote support files:

```text
/u/demistry/p1/check_q6k_loader.py
/u/demistry/p1/sweep_q6k.py
/u/demistry/p1/proxy_ab_q6k.py
/u/demistry/p1/generate_q6k_tokens.py
/u/demistry/p1/kstat2.py
/u/demistry/logs/p1_q6k_nojit.log
/tmp/bench_q6k_jit_debug.log
```

Remote product files, uncommitted:

```text
/u/demistry/tinygrad-src/tinygrad/llm/kernels/nv_q6k.py
/u/demistry/tinygrad-src/tinygrad/llm/kernels/amd.py
```

## 9. Next step depends on the user’s model decision

### Option A — use the working OBLITERATED quant

1. Start llama-server with the corrected filtered-device command above.
2. POST the numeric prompt-token request and compare the 16 raw token IDs.
3. If identical, commit only `nv_q6k.py` and `amd.py` using the project’s `git-personal`
   workflow.
4. Update `progress.md` with the loader-bug disproof, sweep, A/B, benchmark, and profile results.

### Option B — require the Unsloth UD Q4_K_M quant

1. Implement **both** Q3_K `(256, 110)` and IQ4_NL `(32, 18)` in `gguf.py`.
2. Validate each dequant numerically against independent llama.cpp reference ports on real blocks.
3. Load the whole Unsloth model in tinygrad and rerun greedy token generation.
4. Run llama.cpp on the same Unsloth file and compare raw token IDs.
5. Commit loader support separately from the Q6_K NVIDIA kernel/routing change.

Do not silently fall back to OBLITERATED: the user explicitly asked to stop and reconsider after
learning how different the Unsloth quant is.
