# The custom NV kernels for Qwen3.8-27B on tinygrad — a human-readable explanation

Companion to `progress.md`. This document explains what the two custom kernels built in
`tinygrad/llm/kernels/` (branch `qwen27b-nv-q8-kernel`, base `d851aca9a`) actually are, what they
mean, and how they work. The kernel sources are vendored in this repo under `kernels/` so they
survive even if the tinygrad fork is rebased or dropped; `kernels/amd-routing.patch` is the diff
to the pre-existing routing file.

## Why custom kernels at all

The task was: run Qwen3.8-27B (a hybrid `qwen35` architecture — 48 GatedDeltaNet/SSM blocks +
16 attention blocks) on an NVIDIA L40S through tinygrad, and make decode fast.

The generic tinygrad path decoded at **2.06 tok/s** (~59 GB/s effective memory bandwidth) on the
Q8_0 file. Decode (one token at a time) is a **GEMV**: each output neuron is one dot product of a
weight row with the hidden-state vector. This shape is *purely memory-bandwidth-bound* — every
weight byte must be streamed from VRAM exactly once per token, so the best possible throughput is

    max tok/s ≈ memory bandwidth / bytes of weights read per token

The L40S peaks at 864 GB/s. tinygrad's generic matmul codegen — dequantize-heavy, poorly
scheduled access patterns — achieved ~59 GB/s, leaving ~93% of the machine on the table. The fix
is to hand-write the two kernels at tinygrad's **UOp level**: kernels are built as trees of IR
nodes (loops via `UOp.range`, register placeholders, `Tensor.custom_kernel` for outputs) and the
CUDA backend renders them straight to PTX. Raw hardware intrinsics are injected with
`Ops.CUSTOMI`, which emits verbatim C into the rendered source. That is how the kernels get
`__dp4a`, `__shfl_xor_sync`, and `__ldcs` — there is no higher-level tinygrad op for them.

The whole design goal in one sentence: **load every weight byte exactly once, with aligned
vectorized loads, do the dot products as free integer instructions, and reduce with register-only
warp shuffles.**

## Kernel 1 — Q8_0 GEMV (`kernels/nv.py`)

### The layout math

GGUF Q8_0 stores 32 weights per block as a fp16 scale `d` (2 bytes) + 32 int8 quants = **34
bytes**. The kernel does not dequantize through float: it re-views the packed VRAM buffer as
`uint16` — 34 B = **17 words** (`Q8_U16_WORDS = 17`):

| word | content |
|---|---|
| 0 | block scale `d`, bitcast fp16 → float |
| 1..16 | 32 int8 quants, two per uint16 |

Viewing the existing buffer (instead of a lazy bitcast) matters: a bitcast would decompose into
byte-combining ALU and copy the entire 29 GB of weights into every JIT graph.

### Part A — `nv_q8_quantize` (quantize the activation)

The dot products want integers, but the activation `x` is fp32. One warp handles one 32-element
group:

1. warp-shuffle max-reduce of `|x|` → group scale `s = max/127`;
2. each lane quantizes its 4 elements, `round(x/s)` clipped to `[-127, 127]`, and packs 4 int8s
   into one `uint32` word.

Outputs: `xq` (int8, 4 per word) and `xd` (one fp32 scale per group).

### Part B — `nv_linear_q8_0` (the GEMV itself)

Grid: `tokens × out_features × chunks`; **one warp computes 32 groups × 32 inputs = 1024 inputs
of one output neuron** (one "chunk"). If `in_features > 1024` there are several chunk blocks per
neuron and a final reduction over the `chunks` dim.

Per lane, for its group:

1. `_load_lanes(..., 8)` issues a vectorized load: 8 × `uint32` activation words (32 B) and 8
   `uint16` weight words (16 B) per group.
2. Weight words go through a streaming load:

   ```c
   __ldcs((const unsigned short*)ptr)   // ld.global.cs — evict-first, skip L2 allocation
   ```

   Weights are read once per token and never touched again, so allocating L2 for them only evicts
   useful lines. This one hint took Q8_0 from 19.3 → 20.9 tok/s.
3. Two consecutive `uint16` words are combined into one `uint32` holding 4 int8 weights, then:

   ```c
   dot = __dp4a((int)word, (int)xword, dot);
   ```

   `dp4a` is one SASS instruction doing **four** integer byte×byte multiply-accumulates. 8 dp4as
   per lane per group. FLOPs are not the point — the integer dot is nearly free; the kernel exists
   to keep the memory pipe saturated with minimal instructions.
4. Per-group value: `dot × xd[group] × float(bitcast(d))`.
5. `_warp_reduce` (offsets 16, 8, 4, 2, 1 of `__shfl_xor_sync`) sums the 32 lanes **in
   registers** — no shared memory, no atomics. Lane 0 stores the chunk partial.

## Kernel 2 — Q4_K GEMV (`kernels/nv_q4k.py`)

### The layout math

A Q4_K superblock holds 256 weights in **144 bytes**:

| bytes | content |
|---|---|
| 0–1 | `d`, fp16 global scale |
| 2–3 | `dmin`, fp16 global min scale |
| 4–15 | 8 six-bit sub-block scales, packed |
| 16–27 | 8 six-bit sub-block mins, packed |
| 28–155 | 128 bytes of 4-bit nibbles (2 weights/byte) |

144 = **36 uint32**, so the buffer is viewed as `uint32` and every load is naturally 4-byte
aligned — plain loads, no alignment hazards.

The 256 weights are split into **8 sub-blocks of 32**, each with its own 6-bit `scale` and 6-bit
`min`. Per weight the dequantization is

    w = d · scale · q − dmin · min

where `q` is an *unsigned* 4-bit value in `0..15` (an early test-harness reference wrongly used
signed `q−8`; that was a harness bug, not a kernel bug).

Six-bit packing: 8 × 6 bits = 48 bits = 12 bytes, so values straddle byte boundaries. Sub-blocks
0–3 take the low 6 bits of bytes 4..7; sub-blocks 4–7 take the low 4 bits of bytes 8..11 with the
high 2 bits spliced in from the top of bytes 4..7:

    scale  = (sub < 4) ? byte[4+sub] & 63
                       : (byte[8+sub-4] & 15) | ((byte[4+sub-4] >> 6) << 4)

(same pattern for mins on bytes 8..11 / 4..7).

### The qsum trick

The dequant formula has two terms, and the second one (`dmin·min·x_i` per weight) looks like it
needs per-weight work. It doesn't: `Σ (dmin·min)·x_i = dmin·min · Σ x_i`. So the kernel runs
**two** integer dot chains over the same activation words:

```c
dot  = Σ __dp4a(weight_nibbles, xword)   // Σ q_i · x_i
qsum = Σ __dp4a(0x01010101,     xword)   // Σ 1   · x_i  = just Σ x_i
```

The nibble words come in sub-block pairs sharing one `uint32` (low nibble = sub-block 2k, high =
2k+1):

```c
word = (raw[qs_base + i] >> ((subgroup & 1) * 4)) & 0x0f0f0f0f   // 4 unsigned nibbles
```

and the final value is

    (dot · d · scale − qsum · dmin · min) · xd[group]

One extra dp4a chain buys out the entire min term.

### Shared skeleton

Activation quantization is the *same* `_q8_quantize` as the Q8_0 path, and the
token/output/chunk grid, warp reduction, and chunk-partial output are literally shared: `nv_q4k.py`
imports `_decode_linear`, `_nv_dp4a`, `_q8_quantize` from `nv.py` and supplies only its own
`group_dot`. One layout gotcha that cost debugging time: GGUF tiles weights **per output row** —
all 256-weight blocks of neuron `output` are contiguous — so the block base is
`(output · in_features/256 + block) · 36`, not a flat block-major index.

## How they plug in — routing (`kernels/amd-routing.patch`)

`tinygrad/llm/kernels/amd.py` is the pre-existing custom-kernel `Linear` (written for AMD RDNA3).
The branch extends it:

- **`set_quantized`** (lazy, fires on first `__call__`): walks the weight tensor's UOp graph,
  finds the packed `uint8` buffer still sitting in VRAM (the loader keeps weights packed and
  dequantizes lazily), and checks the size signature — `numel/256 × 272` → Q8_0 (272 = 8 blocks ×
  34 B per 256 weights) or `numel/256 × 144` → Q4_K. On NV it claims only those two types and
  re-points `self.weight` at a `uint16`/`uint32` **view** of the same buffer (no copy).
- **`__call__`** then routes by `ggml_type`: Q8_0 → `q8_0_linear`, Q4_K → `q4_k_linear`.
  Symbolic token counts (beam search / chunked prefill) are handled with `pad_to` + `shrink` so
  the kernels see static shapes.
- **Device gating**: AMD/RDNA3 keeps its existing quant claims; NV claims only Q8_0 + Q4_K; other
  devices claim nothing. This also fixed a pre-existing stock-tinygrad crash: K-quants on NV used
  to get packed and then fall through to the generic matmul with a flat 1-D weight → transpose
  IndexError. That is why only the Q8_0 27B file ever loaded before this work.

## Verification

- **Exactness vs numpy**: Q8_0 24/24 small + 4/4 production shapes, `maxerr = 0`. Q4_K 8/8 sweep
  (out 1..128, in 256..2048, partial last warp), relative error ≤ 1.8e-06 — that residual is
  float32 accumulation-order noise; the integer dots are bit-exact.
- **Token-level A/B** on a small proxy model (qwen3.5, temperature 0): 32/32 identical tokens
  with custom kernels engaged (187/187 `Linear`s for Q8_0; 440 `nv_linear_q4_k` launches observed
  in the DEBUG=2 log for Q4_K) vs 0 in the generic path.

## Results

| model / file | generic | custom | llama.cpp (same box) |
|---|---|---|---|
| 27B Q8_0 Uncensored | 2.06 tok/s | **20.9 tok/s** (9.4×, ~616 GB/s) | 22.5 (37.7 with +MTP) |
| 27B Q4_K_M OBLITERATED | 2.56 tok/s | **12.87 tok/s** (5.03×) | not yet measured on this file |

Per-step breakdown of the Q4_K_M custom run: of 77.1 ms/step, GEMV is 20.2 ms. The remaining ~57
ms is the sequential GatedDeltaNet chain + generic Q6_K Linears + launch overhead — the next
bottleneck, not the GEMV.

## Bugs learned the hard way

1. **Gated store (Q8_0)**: gating the output store to lane 0 made tinygrad's gater predicate the
   whole warp → shuffle results undefined. Fix: all 32 lanes active, write a 32-lane scratch
   tile, select lane 0 after the fact.
2. **Double-evaluated shuffle**: `UOp.maximum` renders to the ternary `(a < shfl) ? shfl : a`,
   which evaluates `__shfl_xor_sync` **twice** → garbage (alternating 0/32 in the max-reduce).
   Fix: a single `fmaxf` via `CUSTOMI`. Diagnosis path: sweep 24/24 failed → stage isolation →
   max-reduce broken while sum-reduce fine → DEBUG=4 source dump showed the ternary.
3. **Alignment**: unaligned 4-byte loads and misaligned `__ldcs` hang the kernel (PTX alignment is
   a hard requirement). Hence the 17-word / 36-word buffer views.
4. **The Q4_K "blocker" was a test-harness bug**: the A/B counter walked dicts but not lists;
   `Transformer.blk` is a plain list, so block Linears were never counted and engagements read as
   0. Detection and routing were fine all along. Fixed in place.

## What's next (ranked, from `progress.md`)

- **P0** — profile the ~57 ms/step non-GEMV time before touching anything.
- **P1** — DeltaNet chain (biggest lever, expected 1.5–2×): the sequential GatedDeltaNet
  recurrence (conv1d + state update) dominates decode now.
- **P2** — NV chunked prefill (the AMD-only gate at `llm/model.py:479` forces `chunk_size=1` on
  NV → token-by-token prefill; llama.cpp does ~2600 tok/s on pp512).
- **P3** — Q4_K GEMV micro-opts (`__ldcs`, 16 B vectorized loads, packed scale words) only if P0
  says GEMV still matters.
- **P4** — Q6_K NV kernel (the remaining generic Linears in Q4_K_M/Q6_K files).
- **P5** — BEAM_CACHE validation (3.4–3.7× headroom measured on 0.8B generic ops, unverified).
- **P6** — speculative decoding / MTP (last structural gap vs llama.cpp's +MTP 37.7 tok/s).
- **P7** — hygiene: llama.cpp benchmark on the Q4_K_M file, Q3_K/Q8_K loader types, etc.

On MTP specifically: the checkpoints carry one extra MTP block beyond the 65 main blocks
(the 65+1 accounting, keyed by the `qwen35.nextn_predict_layers` metadata). Today tinygrad
strips it: `model.py` subtracts that count from `block_count` when building the main
`Transformer`, and the leftover MTP tensors are dropped by `load_state_dict(consume=True)`.
MTP is implementable (keep the MTP block weights, run it as a 1-token draft after each
sampling step, verify the draft with a batched forward pass) but it needs the draft+verify
loop in `generate()` — including a shadow copy of the GatedDeltaNet recurrent state for the
draft step — and the verify pass wants chunked prefill on NV first (P2), which is why the
plan ranks it P6 behind P0–P5.
