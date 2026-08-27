# Session 5+ Consolidation — Q6_K NV Kernel Verified; Unsloth Dynamic Quants Planned (2026-08-27)

## Critical Corrections from p1-q6k-session5-handoff.md

### 1. The Q6_K loader "bug" was FALSE — DO NOT apply any `gguf.py` shift fix
The session 4 assumption (`xl | xh` missing `<<4`) was incorrect. `xh` is already shifted at its definition (`q_to_uint8(..., 2).lshift(4)`, commit `67ed4c4eb3`). The loader is correct. The incorrect fix (`8047d69b8`) was applied and then **reverted** (`65e09942f`). The corrected `check_q6k_loader.py` confirms exact match on real model bytes.

### 2. "Vanishing model" was a filename typo
Actual file: `/scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (hyphen, not dot). Stable copy: `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (`16,810,705,952` bytes).

### 3. nv_q6k.py was verified and committed this session
Commit `1fe4ba369` (`nv_q6k.py` + `amd.py` routing). Sweep (`sweep_q6k.py`): `7/7 OK`, max rel error `1.07e-07`. Profile (`kstat2.py`): `65` calls/step, `7.4 ms` (vs `51.4 ms` generic Q6_K `65%` → now `19.4%`). Benchmark: `28.5 tok/s` (`2.3×` over `12.49 tok/s`). Decode graph: `39.74 ms/step` (`2×` faster vs `78.84 ms`).

### 4. User switched to Unsloth Dynamic v3.0 quant model
New file: `/scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf` (`16,464,440,224` bytes). Quant mix is very different from OBLITERATED:
```
F32(360) Q8_0(106) Q3_K(7) Q4_K(104) Q5_K(131) Q6_K(30) IQ4_NL(7) IQ3_S(4) IQ4_XS(117)
```
Tinygrad fails immediately: `ValueError: GGML type '11' is not supported!` (Q3_K). Also requires IQ4_NL (type 20), IQ3_S (21), IQ4_XS (23), Q5_K (13). These are separate loader features requiring reference tests.

## Note: "We won't be working with [Q6_K loader fix] anymore"
The incorrect loader fix (`8047d69b8`) has been reverted. No further loader work needed for Q6_K — it was always correct. The focus shifts to:
1. Unsloth Dynamic v3.0 quant support (new GGML types + kernel routing)
2. The small `gemma-3-270m-it` model for kernel development

---

## New Plan: Unsloth Dynamic Quant Kernels (per user request 2026-08-27)

### Reference docs
- Unsloth Dynamic v3.0: https://unsloth.ai/docs/basics/dynamic-3.0-ggufs
- Unsloth Dynamic v3.0 GGUFs: https://huggingface.co/unsloth/gemma-3-270m-it-GGUF
- Gemma 3 model (small, for development): `unsloth/gemma-3-270m-it-GGUF`

### Download command (run from `/scratch/local/demistry/models/`)
```bash
hf download hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-UD-Q8_K_XL.gguf --local-dir .
```
Note: `node-lair` has `hf` authenticated. The download target is the small `gemma-3-270m-it` checkpoint (`UD-Q8_K_XL`), not the full 27B.

### Required loader additions (separate feature, per session-5 handoff)
- **Q3_K** (GGML type 11): 256 elements, 110 bytes (`hmask[32]`, `qs[64]`, `scales[12]`, `d:f16[2]`). Reference: `ggml/src/ggml-quants.c::dequantize_row_q3_K`.
- **IQ4_NL** (GGML type 20): 32 elements, 18 bytes (`d:f16[2]`, `qs[16]`). Uses existing `kvalues_iq4nl` nonlinear LUT (`amd.py` line 99).
- **IQ3_S** (GGML type 21): 32 elements per subgroup.
- **IQ4_XS** (GGML type 23): already partially supported? Check loader.
- **Q5_K** (GGML type 13): partially supported? Check.

These must be added as independent loader features with reference tests, NOT incidental to Q6_K kernel.

### Kernel plan (new task, not P1-P7 from original ask)
1. **Loader**: Add Q3_K, IQ4_NL, IQ3_S, Q5_K to `gguf.py` with reference ports from `llama.cpp/ggml-quants.c`.
2. **Routing**: Add device-gated claims for new quant types in `amd.py` (NV path).
3. **Custom GEMV kernels** (optional, based on profile needs): If the new model uses Q3_K/Q5_K extensively and the generic matmul is slow, build NV GEMV kernels mirroring `nv_q4k.py`/`nv_q6k.py`.
4. **Verification**: Sweep against `llama.cpp` scalar dequant + numpy reference matmul, then small-model proxy A/B (`gemma-3-270m-it` greedy decode, `32/32` token match).

### Next concrete step (pending user approval)
Download the `gemma-3-270m-it` checkpoint to verify the file is readable and inspect its GGUF quant mix before designing loader additions.
