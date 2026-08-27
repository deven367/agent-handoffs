# Handoff — Unsloth NaN Blocker RESOLVED; verify + remaining optimizations (2026-08-27, take 2)

## Executive state

**The NaN blocker is FIXED.** Full custom-kernel logits on the Unsloth `UD-Q4_K_M` model are now finite and match generic (`argmax` identical 5328, corr 0.99989). Root cause: **byte-count routing collision** between `IQ4_NL` (18 B / 32 elems = 0.5625 B/elem) and `Q4_K` (144 B / 256 = 0.5625 B/elem). The old `set_quantized` heuristic keys packed sizes solely on total bytes, so 7 IQ4_NL tensors were misclaimed as Q4_K, fed to the Q4_K GEMV kernel, and produced garbage → cascade NaN.

Fix: thread the **actual GGUF tensor types** through the loader into every `Linear`, replacing the byte heuristic for packed claims.

Token-level A/B (custom vs generic decode) was launched but **canceled at session end** — rerun it to complete end-to-end identity proof.

No GPU process left running.

## Fix (uncommitted, must be preserved)

Three files on `node-lair:/u/demistry/tinygrad-src` (branch `qwen27b-nv-q8-kernel`, ahead 6):

```text
 M tinygrad/llm/gguf.py
 M tinygrad/llm/kernels/amd.py
 M tinygrad/llm/model.py
?? tinygrad/llm/kernels/nv_iq4xs.py
?? tinygrad/llm/kernels/nv_q5k.py
```

1. **`gguf.py`** — `_gguf_parse` now stores `kv_data['tinygrad.tensor_types'] = {name: ggml_type}` (per-tensor type side-channel).
2. **`model.py`** `Transformer.from_gguf` — after `load_state_dict`, walks all blocks and stamps `Linear.ggml_type` from `tinygrad.tensor_types`; sets `_needs_pack=True` when the type is NV-custom-supported (8,12,13,14,23). This runs only when `tinygrad.tensor_types` exists (loader-provided).
3. **`amd.py`** `Linear.__call__` — if `_needs_pack and nv_supported`: call `set_quantized` then mark done. `set_quantized` claims NV types by element count (`numel // elems * bytes`) — the collision disappears because the packed type is **stamped**, not guessed.

Note: `use_custom_quant=False` (generic mode) still works — pack is gated on `nv_supported` which includes `use_custom_quant`.

## Verification results (measured)

| check | result |
|---|---|
| loader exactness (Q3_K, IQ4_NL) | `maxabs=0 rel=0 exact=True` (random + real blocks) |
| Q5_K sweep 5 shapes | all `rel ≤ 1.25e-7` |
| IQ4_XS sweep 5 shapes | all `rel ≤ 8.7e-8` |
| full-model custom logits | finite, argmax 5328, corr vs generic `0.99989`, maxabs `0.151` |
| blocks 0–1 chain | finite both modes |

Scripts:
- `/u/demistry/p1/check_unsloth_loaders.py` (loader exactness)
- `/u/demistry/p1/sweep_nv_unsloth.py` (Q5_K / IQ4_XS sweeps)
- `/u/demistry/p1/unsloth_logits.py custom <out.npy>` / `generic <out.npy>` (full-logit A/B)
- `/u/demistry/p1/ud_ab.py <custom|generic> <model> <n_tok>` — token A/B + decode speed (canceled mid-run)

## Pending work (in priority order)

1. **Finish token A/B + decode bench on the Unsloth Q4 file** (custom vs generic). Then the meaningful number: `benchmark` path = `python3 -m tinygrad.llm --model ...UD-Q4_K_M.gguf --max_context 512 --benchmark 15`. Also grab per-step GPU time (`*** NV ... batched` line) and compare to prior `28.5 tok/s` (Uncensored file) — this file is different: Q6_K count lower (30 vs 67), but Q5_K 131 + IQ4_XS 117 are the bulk, now custom.

2. **llama.cpp same-file comparison** — `llama-bench` or a noninteractive `llama-server --n-gpu-layers 999`. Model is the **same** `Qwen3.8-27B-UD-Q4_K_M.gguf` (~16.5 GB). Report `pp`/`tg` separately. Prior attempts via interactive `llama-cli` hung (no TTY); use `--batch-size 2048` etc. or `llama-bench -m ... -p 512 -n 128`.

3. **Validate chunked prefill gate removal (`29a306ec6`)** — chunked prefill on NV with the new stamping; compare token sequence 1:1 vs `chunk_size=1`, and prefill ms.

4. **MTP / other NV kernels** — later. DeltaNet, BEAM_CACHE, MTP remain.

## Gotchas / notes

- Do **not** benchmark or serve the Unsloth model under `make serve-tg` until tokens A/B identical; serve target points at Uncensored file (fine).
- `/u/demistry/p1/inspect_gguf.py` enum-name table is stale for types ≥ 15; trust `tinygrad.tensor_types` or `llama.cpp ggml.h`.
- The claim heuristic in `set_quantized` is now: `packed_blocks = {Q8_0: (272,256), Q4_K: (4*4,256), Q5_K: (40*4,256), Q6_K:(210,256), IQ4_XS:(34*4,256)}` — keys by `decoded.numel() // elems * bytes`.
- Sweeps expect a specific LUT layout for IQ4_XS (`kvalues_iq4nl`); the CUDA `__byte_perm` approach was abandoned in favor of a tiny lazy int8 LUT buffer (no `.realize()` inside `@function`).

## Suggested first commands

```bash
ssh node-lair
cd /u/demistry/tinygrad-src
git status --short --branch
python3 /u/demistry/p1/ud_ab.py custom /scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf 12
python3 /u/demistry/p1/ud_ab.py generic /scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf 12
diff /tmp/ab_custom.txt /tmp/ab_generic.txt
```

Then run the full benchmark (task 1) and llama.cpp comparison (task 2).

## Delivery state

- All fixes **uncommitted**; do not lose `nv_iq4xs.py`, `nv_q5k.py`, and the three modified files.
- No commit, no push this session.
- Local docs repo (`agent-handoffs`) has uncommitted files incl. this handoff.