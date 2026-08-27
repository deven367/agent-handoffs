# P1 Handoff — Qwen3.8-27B tinygrad decode optimization (session 3, 2026-08-26)

**Status: P0 profiling COMPLETE. The P1 target is now identified and differs from the
session-2 plan's assumption. Ready to build: `nv_q6k.py` (Q6_K GEMV kernel) — see §6.**

Read `progress.md` first (sessions 1–2, both custom kernels + verification method).
This doc is self-contained for the P1 build.

---

## 1. Environment — Slurm node gotchas (cost hours last session)

- `node-lair` is a **Slurm node** (job 85895, `sbatch`). **`/tmp` is wiped** — put all
  scratch under `/u/demistry/p1/` (scripts already there) and logs under `/u/demistry/logs/`.
- 8 physical L40S on the box; this job is allocated **2** (only what NVML sees):
  - `nvidia-smi` index 0, bus `0000:41:00.0`, **device minor 1** — a `llama-server`
    (40 GiB). **Never touch.**
  - `nvidia-smi` index 1, bus `0000:61:00.0`, **device minor 0** — free. **This is ours.**
- tinygrad's NV runtime indexes devices **by device minor number, not smi index**
  (enumerates via `NV_ESC_CARD_INFO`, `dev_ids` = minor numbers):
  - `gpus_info[0]` = minor 0 = smi index 1 = **the free GPU**.
  - `gpus_info[1]` = minor 1 = smi index 0 = the busy one.
  - So **default `Device.DEFAULT` (NV, device_id 0) already lands on the free GPU.**
    `DEV=":1+NV"` is the busy GPU — **do not use**. (DEV syntax: `IFACE:INDICES+DEVICE:RENDERER`.)
  - Sanity check that you're on the free GPU: the 17 GB model load succeeds (busy GPU has
    5.7 GB free → immediate OOM).
- Baseline reproduced this session (free GPU, JIT on): **12.49 tok/s, 80.08 ms/step**
  (session-2: 12.87 — noise).

## 2. How the bench number is composed (do not be fooled)

`python3 -m tinygrad.llm` bench loop calls `generate()` **once per measured step**, so each
step re-runs the full prefill (6 chunk graphs: 32/64/128/256/512/1024) + 1 decode token:

| phase | GPU time/step |
|---|---|
| prefill chunks (6 JIT graphs) | ~57.3 ms (26.1 + 13.6 + 11.3 + 6.1 + 0.12 + 0.05) |
| decode graph (266 kernels) | **20.3 ms** |
| **total (the 12.49 tok/s metric)** | **~77.8 ms** |

The session-2 plan's "GEMV 20.24 ms of 77.09 ms" was this decode graph.
**Real decode speed is ~49 tok/s today (20.3 ms/token), not 12.5.** When benchmarking P1,
compare the decode-graph line (`*** NV ... batched 266 ... tm Xms`) or run a
decode-only harness; the prefill part is invariant to GEMV-kernel changes.

## 3. P0: per-step profile (the actual P1 target)

From `/u/demistry/logs/p1_nojit.log` (JIT=0 DEBUG=2, 10 bench steps, 2282 kernels/step,
GPU-busy 78.84 ms/step; parsed with `/u/demistry/p1/kstat2.py` — segments on
`GlobalCounters` resets, see cli.py:180):

| kernel family | count/step | ms/step | share | identity (confirmed §4) |
|---|---|---|---|---|
| `r_40_32_4_68_2_2_2_32` | 32 | 33.2 | 42% | **ffn_down Q6_K GEMV** (17408×5120) ×33 blocks |
| `nv_linear_q4_k` | 432 | 18.9 | 24% | custom Q4_K GEMV (already fast, 43.6 µs/call) |
| `r_80_32_4_20_2_2_2_32` | 24 | 11.7 | 15% | **attn_qkv Q6_K GEMV** (10240×5120) ×24 SSM blocks |
| `r_1940_32_4_20_2_2_2_32` | 1 | 4.5 | 6% | **output/lm_head Q6_K GEMV** (248320×5120); 1940 = 248320/128 chunks |
| `r_4_2_8_16_2_20_2_2_2_32` | 8 | 2.0 | 2.5% | **attn_v Q6_K GEMV** (1024×5120) ×9 attn blocks |
| `r_3_2_2_8_16_4_32_4` | 96 | 4.6 | 6% | **SSM state mat-vec** (48 blocks × {state@k, state@q}), 48 µs avg |
| `nv_q8_quantize` + small `r_*`/`E_*` | ~1700 | ~4 | 5% | activation quant + norms/casts/conv/attention-KV |

**Headline: the session-2 assumption was wrong.** The SSM/DeltaNet chain is only
~4.6 ms/step (6%), not the ~57 ms hypothesized. **The dominant cost is the 67 Q6_K
linears running the generic matmul path on NV: ~51.4 ms/step = 65% of GPU time.**
Per-call times are 5–35× over the memory bound (e.g. ffn_down: 1.04 ms for 73 MB of
weights ≈ 70 GB/s effective).

## 4. Evidence chain (how the mapping was done — repeatable)

1. GGUF tensor census (`/u/demistry/p1/inspect_gguf.py` on
   `/scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`, 866 tensors):
   - **Q4_K: 439** — all other linears (attn_gate, ssm_beta/alpha, ssm_out, ffn_gate/up…)
   - **Q6_K: 67** = `output.weight` (1) + `ffn_down` (33 blocks) + `attn_qkv` (24 SSM
     blocks) + `attn_v` (9 attn blocks)
   - F32: 360 (norms, ssm_a/dt, conv1d, rope-freq — not GEMVs)
2. NV routing (`amd.py` `set_quantized`) only claims **Q4_K and Q8_0** on NV; Q6_K has no
   NV kernel → generic matmul. Claimed Q4_K GEMVs: 65 blocks × 7 + lm... = **432/step**,
   matching the `nv_linear_q4_k` count exactly (48 SSM × 7 + 16 attn × 6.5… = 432 ✓).
3. Isolated 2-block run (`/u/demistry/p1/map_kernels.py`: 1 GatedDeltaNet + 1 attention
   block, exact 27B shapes, fp32 random weights, 3 decode forwards, DEBUG=2 →
   `/u/demistry/p1/map_run.log`): the big generic GEMVs emit names with the same leading
   patterns as the model profile (`r_40_32_4_*`, `r_80_32_4_*`), and the SSM state
   mat-vec emits `r_3_2_2_8_16_4_32_4` at ~30 µs — matching the model's 48 µs family.
   (fp32 weights ⇒ GEMV names' tail dims differ from the quantized run; match on leading
   pattern + count, not full name.)
4. Counts line up: 33/24/9/1 Q6_K tensors ↔ 32/24/8/1 profile families (±1 =
   JIT/fusion or window edge).

## 5. What is NOT the bottleneck (measured, closed)

- SSM state ops: 96 reduces @ 48 µs = 4.6 ms/step (6%). Fusing (plan P1 as originally
  written) still worth doing for kernel-count/launch overhead + MTP, but it is ~1/11th of
  the Q6_K opportunity.
- Attention KV-cache reduces (symbolic `start_pos+1` shapes): ~0.13 ms/step total.
- nv_q8_quantize: 0.57 ms/step.
- Non-JIT wall (1103 ms/step) is DEBUG=2 per-kernel-sync artifact; JIT wall ≈ GPU busy.

## 6. P1 build: `nv_q6k.py` — Q6_K decode GEMV (highest ROI, ~51 ms/step available)

**Structure: copy `tinygrad/llm/kernels/nv_q4k.py` → `nv_q6k.py`, swap only `group_dot`.**
The skeleton (q8_quantize reuse, warp-per-1024-inputs, chunk grid, `_decode_linear`,
routing, `KernelInfo`, exact-store) is already proven — session 2's Q4_K kernel is the
template. Q6_K block = 256 weights / **210 bytes** (odd → view as `uint8`, or `uint16`
where 2-byte aligned; block stride 210 ⇒ 2-byte aligned, 4-byte NOT guaranteed — use
16-bit or 8-bit loads, no `__ldcs` uint32 unless you prove alignment).

**Q6_K dequant (verified against working AMD reference, `amd.py:192-203`):**
per 256-weight block, byte offsets from block base:
- `ql[0:128]` — low 4 bits of 256 weights (byte j ↔ weights 2j, 2j+1; first 64 bytes =
  first 128 weights, next 64 = last 128)
- `qh[128:192]` — high 2 bits (4 weights/byte)
- `scales[192:204]` — 12 int8 scales; subgroup s (of 8, 32 weights each) uses
  `scales[2s]` for weights s·32..s·32+15 and `scales[2s+1]` for s·32+16..s·32+31
- `d[208:210]` — f16 (uint16) global scale; bytes 204–207 (h/pad) unused for dequant

For weight at position `pos` (0..255) in subgroup `subgroup = pos//32`:
```
q   = ((low_nibble(pos) | (high_2bits(pos) << 4))) - 32      # int8, -32..31
val = q * scales[2*subgroup + (pos%32)//16] * d
row = Σ_pos val * x[pos] * xd[group]                          # same xd as Q4_K path
```
Integer core: per 32-weight group, 2 dp4a chains (16 weights each) with `q+32` bias-free
repacking: pack 4 × int8 into uint32, `__dp4a` against the x-words (same xq/xd as the
Q4_K kernel — **reuse `_q8_quantize` verbatim**). Then
`(dot0*s0 + dot1*s1) * xd * d`. Expect ~16 dp4a + 11 uint16 loads per group (vs 8/8 Q4_K).

**Expected impact (memory-bound estimate, ~600 GB/s like the Q4_K kernel achieved 616):**
ffn_down 73 MB → ~120 µs (now 1040), attn_qkv 42.5 MB → ~70 µs (now 490),
lm_head 1.03 GB → ~1.7 ms (now 4550), attn_v 4.2 MB → ~7 µs (now 244)
⇒ Q6_K total ~51.4 → **~7.5 ms/step**; decode graph ~20.3 → ~13–14 ms/step (≈2.4× real
decode throughput).

**Routing change (`amd.py`):** in `set_quantized`, on NV add Q6_K → `q6_k_linear` claim
(reinterpret the SHINK uint8 buffer as uint16/uint8 view — the Q8_0 pattern,
`nv.py` ~line 100: `raw.buf_uop` re-viewed by word size; check `numel // 256 × 210`).
Keep the AMD path and the Q4_K/Q8_0 claims untouched.

**Verification (reuse session-2 harness pattern — it is in git history / progress.md):**
1. numpy reference sweep: random weights, several (out_features, in_features) incl.
   17408×5120, 10240×5120, 248320×5120, 1024×5120 + odd in_features padding case;
   assert rel-err ≤ ~1e-5 (f32 accumulation-order noise) on dequant AND full GEMV.
2. Token A/B on a small model (session-2 proxy method): 32/32 identical tokens, custom
   vs generic; plus DEBUG=2 engagement count (expect 67 `nv_linear_q6` calls/step).
3. Full 27B: decode-graph `tm` from the `*** NV ... batched` line, pre- vs post-
   (ignore the prefill-dominated tok/s number, §2).

## 7. P2 (after Q6_K): fused SSM decode kernel — design already sketched

One warp per (v_head, v_row): lane l covers k-cols [16l..16l+15] of the 128×128 state
tile; shared `s1 = state*alpha` read; `qsum = Σ_k state[l,k]·x_k` via 2×dp4a against
int8-quantized x (or plain f32 mul-add — state tiles are small); `v_new = v −
(s1@k)·beta`; rank-1 update; `o = (s1 + v_new⊗k) @ q`; f16 state store. Replaces
~10 kernels × 48 blocks/step (4.6 ms GPU + launch overhead). Design detail: the
q8-quantized-x trick lets the two mat-vecs share one int8 dot path; keep f32 accum.
Files: new `gated_delta_decode` in `nv.py` (or `nv_ssm.py`), route from
`GatedDeltaNetBlock._attention` when `nv_custom_kernels_supported` (currently the SSM
path has **no** custom route on NV — it runs generic; the `flash_attention` custom path
is AMD-only, model.py:187).

## 8. MTP (context, not P1)

`nextn_predict_layers = 1` confirmed (GGUF metadata; `nextn.*` tensors present;
llama.cpp `--draft-mtp` uses them; 37.7 vs 20.9 tok/s with it). Implementable in
tinygrad (loader strips it today) but low ROI vs §6 — defer.

## 9. Artifacts

| what | where |
|---|---|
| tinygrad work tree | `node-lair:/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`, HEAD `2d46ea489` (no changes made this session) |
| custom kernels (reference templates) | `tinygrad/llm/kernels/{nv.py, nv_q4k.py, amd.py}` |
| P0 profile logs | `/u/demistry/logs/{p1_dbg.log (JIT), p1_nojit.log (JIT=0)}` |
| scripts (persistent dir) | `/u/demistry/p1/{kstat2.py (log parser), map_kernels.py (2-block mapping run), inspect_gguf.py (tensor census)}` |
| mapping run output | `/u/demistry/p1/map_run.log` |
| model file | `/scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (17.04 GB) |
| commit identity | `~/bin/git-personal` wrapper, `deven367 <masterdeven@gmail.com>` |

## 10. Suggested order

1. Build `nv_q6k.py` + routing (§6), verify per §6. Expected: decode 20.3 → ~13 ms.
2. Profile again with `kstat2.py` — confirm Q6_K families gone; next bottleneck will be
   Q4_K GEMV (18.9 ms) and SSM chain (4.6 ms).
3. P2 SSM fused kernel (§7).
4. Optional: Q4_K GEMV micro-opts (the 43.6 µs/call is ~616 GB/s, already near peak).
