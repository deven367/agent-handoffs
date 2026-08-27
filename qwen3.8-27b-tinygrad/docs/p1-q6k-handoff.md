# Session 4 Handoff — Q6_K NV kernel + Q6_K loader bug (2026-08-26, ~23:45 EDT)

**Status: PAUSED on user request.** Q6_K kernel is fully designed and drafted
(`kernels/nv_q6k.py` in this repo — **unverified, never compiled**). Nothing was applied
to the remote tree (still HEAD `2d46ea489`, clean). The big finding of this session:

## 1. BUG: tinygrad's Q6_K loader dequant is wrong (one line in gguf.py)

`tinygrad/llm/gguf.py`, `ggml_data_to_tensor`, `ggml_type == 14` branch, last line:

```python
return d * (xl.bitwise_or(xh).bitcast(dtypes.int8) - 32).flatten(-2) * scales   # BUG
```

`xl[j]` = low 4 bits of weight j (bits 0–3), `xh[j]` = high 2 bits of weight j
(bits 0–1 of a 2-bit value — `q_to_uint8(t, 2)`). ORing them puts the high 2 bits in the
LOW positions: the 6-bit code cannot even be represented. Correct combine (per ggml C):

```python
return d * (xl.bitwise_or(xh.lshift(4)).bitcast(dtypes.int8) - 32).flatten(-2) * scales
```

**Evidence chain (all three independent):**
1. llama.cpp `ggml/src/ggml-quants.c` `dequantize_row_q6_K` (fetched from master into
   local `/tmp/ggml-quants.c`): `q1 = (ql[l]&0xF) | (((qh[l]>>0)&3)<<4) - 32` etc.
2. `amd.py::_quant_decode_kernel` Q6_K branch (remote, lines ~192–207):
   `quant = ((low & 15) | ((high & 3) << 4)).bitcast(dtypes.int8) - 32` — correct combine.
   The AMD custom kernel and the loader DISAGREE; the C reference sides with the AMD kernel.
3. Q5_K in the same loader does `q + qh * 16` (proper shift) — Q6_K's unshifted OR is the outlier.

**Impact:** every Q6_K weight in every model loaded by this tinygrad is corrupted.
On the 27B OBLITERATED Q4_K_M file that's 67 tensors (lm_head, ffn_down×33, attn_qkv×24,
attn_v×9) — the dominant generic-path cost in the P0 profile. Text output from that file
was never compared to llama.cpp (P7 skipped), so this went unnoticed.
Q8_0/Q4_K files are unaffected (no Q6_K branch). The 1-line fix is correct for AMD too
(AMD custom Q6_K kernel already matches C; the fix makes generic match it).

**Not yet numerically confirmed on real bytes** — `scripts/check_q6k_loader.py`
(also at `node-lair:/u/demistry/p1/check_q6k_loader.py`) is ready but the model file was
unavailable for >5 min straight when it ran (see §3). It compares, on real Q6_K blocks:
loader-current vs loader-fixed vs an independent numpy port of the C dequant.
Expected: fixed == C exact, current != C. Run it first thing.

## 2. Q6_K layout (triple-verified; supersedes p1-handoff.md §6 which had errors)

Block = 210 bytes: `ql[0:128]  qh[128:192]  scales[192:208] (16 × int8!)  d[208:210] (f16)`.
(p1-handoff said "12 scales + pad" — wrong: 16 scales, no pad, no dmin field in the file.)

For weight w (0..255) in a block, with h = w//128, m = w%128:
- low 4 bits:  ql byte `h*64 + (m%64)`, high nibble if m >= 64 else low nibble
- high 2 bits: qh byte `128 + h*32 + (m%32)`, 2-bit slot `(m//32)` (bits 2t..2t+1)
- scale: `scales[w//16]` (int8)
- value: `((low | (high<<4)) - 32) * scale * d`

Per 32-weight subgroup s (word_idx w=0..7 covers weights s*32+4w..+3), u16 word offsets
(block = 105 u16 words; 210k is 2-aligned not 4-aligned → **uint16 view only**, never u32):
- ql u16 base `16*(s%2) + 32*(s//4) + 2w` (2 loads → 4 bytes), nibble shift 4 iff s&2
- qh u16 base `64 + 16*(s//4) + 2w` (2 loads), 2-bit shift `2*(s%4)`
- `qword = ((ql >> nib)&0x0F0F0F0F) | (((qh >> qsh)&0x03030303) << 4)` = 4×(q+32) bytes
- two dp4a chains (word_idx 0-3 → scales[2s], 4-7 → scales[2s+1]) + two qsum chains;
  result `((dot0−32·qsum0)·s0 + (dot1−32·qsum1)·s1) · xd · d`
- scale pair = u16 word `96+s`; d = u16 word `104`

Per subgroup: 34 u16 loads + 16 dp4a. Instruction estimate for ffn_down (17408×5120) ≈ 32 µs
vs memory 73 MB ≈ 122 µs at 600 GB/s → still memory-bound like Q4_K (43.6 µs/call @ 616 GB/s).
Reuses `_q8_quantize`, `_decode_linear`, `_load_lanes` verbatim (warp-per-output-row pattern).

## 3. Environment incident — OBLITERATED model file keeps vanishing

`/scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf` is intermittently
UNLINKED and re-created by an unknown process (shared Slurm box, job 85960 is ours):
`ls` shows it, `stat`/`open` give ENOENT; windows of absence last minutes; same size
(16810705952)/mtime (Aug 25 23:51) every reappearance; the other 4 files in the dir are
stable; no visible owning process (`/proc/*/fd` scan, squeue clean).
**Copies to `/data/user/demistry` (user-suggested, 70T NFS, free) failed 8/8** because the
source was absent during each `cp`. The checker's 60×5 s retry also fully missed.
→ **First action next session: poll `stat` until it exists, then `cp` to
`/data/user/demistry/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf` (retry loop, verify size),
and use the COPY for all model runs** (both tinygrad `--model` and llama.cpp).

## 4. What to do next (ordered)

1. **Stabilize the model file** (§3) → `/data/user/demistry/…`
2. **Confirm loader bug**: `python3 /u/demistry/p1/check_q6k_loader.py <copy>`
3. **Apply loader fix** (1 line, §1) — separate commit: "fix Q6_K dequant: shift high 2 bits to bits 4-5".
   Required BEFORE the kernel A/B (custom kernel would mismatch the broken generic path).
4. **Apply kernel + routing**:
   - `kernels/nv_q6k.py` (this repo, unverified draft) → `tinygrad/llm/kernels/nv_q6k.py`
   - `amd.py` changes:
     - import: `from tinygrad.llm.kernels.nv_q6k import q6_k_linear`
     - `set_quantized` NV block: add `packed_sizes[decoded.numel() // 256 * Q6_BYTES] = Q6_K`
     - `set_quantized` view line (currently `packed_dtype = dtypes.uint8 if self.ggml_type == Q6_K else dtypes.uint16 if self.ggml_type == Q8_0 else dtypes.uint32`):
       `nv_dev = nv_custom_kernels_supported(decoded.device)` then
       `packed_dtype = dtypes.uint16 if (self.ggml_type in (Q8_0, Q6_K) and nv_dev) else dtypes.uint8 if self.ggml_type == Q6_K else dtypes.uint32`
       (AMD keeps uint8 Q6_K; NV gets uint16; the existing `raw_offset % itemsize == 0` assert covers alignment)
     - `__call__`: after the `Q4_K and nv_supported` branch, mirror it for `Q6_K` → `q6_k_linear(self, x)` (int numel / pad_to+shrink for symbolic)
5. **Sweep**: `scripts/sweep_q6k.py` (in this repo) — random integer q ∈ [−32,31] + random
   int8 scales + f16 d, packed per §2 layout, kernel vs exact-dequant f32 matmul.
   Shapes: 17408×5120, 10240×5120, 1024×5120, 248320×5120 (lm_head), 768×768 (partial warp),
   256×256, 512×1024; tokens 1–4. Cross-check harness style with `/u/demistry/sweep_q4k.py`.
   Activation trick from session 2: x = k·u with max|k| = 127 per 32-group, u = 2^-10·(1..4)
   → `_q8_quantize` exact.
6. **Proxy A/B**: copy of `/u/demistry/proxy_ab_q4k.py` with a Q6_K counter
   (expect 70 `nv_linear_q6_k`/step on the 4b Q4_K_M model; 32/32 tokens identical).
7. **27B decode-graph bench** (pre = 20.3 ms session 3; post expected ~13 ms):
   `python3 -m tinygrad.llm --model <copy> --max_context 512 --benchmark 20`,
   read the `*** NV … batched … tm` line, not the tok/s line (prefill-dominated, see p1-handoff §2).
8. **Re-profile** `/u/demistry/p1/kstat2.py` on a fresh JIT=0 DEBUG=2 log → Q6_K families gone.
9. **llama.cpp cross-check on the OBLITERATED file** (completes P7 AND proves the loader fix
   end-to-end): user pointed at `/u/demistry/llama.cpp` — **full CUDA build at
   `/u/demistry/llama.cpp/build/bin`** (llama-server, llama-cli, llama-bench; HEAD bf9421646).
   Run llama-cli with the copy, fixed seed + temp 0, same prompt as tinygrad, compare tokens.
   Use GPU1 (free, device minor 0 — tinygrad's default) — sequence it AFTER the tinygrad runs
   (GPU0 has only ~5.7 GB free: Q8_0 llama-server, port 9932, don't touch).
10. Commit (git-personal wrapper), update progress.md.

## 5. Known risks / open questions

- `nv_q6k.py` has never been compiled; UOp API slips possible (mitigation: it mirrors
  nv_q4k.py line-for-line in structure; debug per progress.md gotcha #4 — DEBUG dumps from
  the worker, not the driver).
- The file-recreation process is unidentified. If it keeps the file gone, 27B work blocks;
  sweep + proxy A/B (small model, HF cache) do NOT need it.
- If someone is actively re-downloading the models dir, the copy may race a newer revision —
  accept it, note the mtime in progress.md.
- Loader fix changes generic-path output for ALL Q6_K models (AMD too) — justified by the C
  reference; no AMD GPU here to re-verify (code-review level, per P7).

## 6. Artifacts

Local (this repo): `kernels/nv_q6k.py` (draft), `scripts/check_q6k_loader.py`,
`scripts/sweep_q6k.py`, `kernels/{nv.py,nv_q4k.py,amd-routing.patch}` (session-2 refs;
nv.py & nv_q4k.py VERIFIED identical to remote tree — md5 match, 2026-08-26 session 4),
`/tmp/ggml-quants.c` + `/tmp/ggml-quants.h` (llama.cpp master, dequant reference at
`dequantize_row_q6_K`, line 1939 of the .c).

Remote (node-lair): `/u/demistry/p1/` (kstat2.py, map_kernels.py, inspect_gguf.py,
check_q6k_loader.py), `/u/demistry/sweep_q4k.py`, `/u/demistry/proxy_ab_q4k.py`,
`/u/demistry/llama.cpp` (CUDA build), tinygrad tree `2d46ea489` (untouched this session).

GPU: smi idx 1 / device minor 0 = free (tinygrad default lands here); smi idx 0 / minor 1 =
Q8_0 llama-server 40 GB (untouchable). Slurm job 85960 still running at pause.
