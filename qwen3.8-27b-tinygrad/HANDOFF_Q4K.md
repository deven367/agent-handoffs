# Handoff — Q4_K NVIDIA kernel (updated 2026-08-26 session 2)

## Status: A/B GREEN 32/32 — Q4_K kernel wired, verified, committed, benchmark in progress

### Done and verified this session
1. **Kernel exactness proven.** `python3 /u/demistry/sweep_q4k.py` -> 8/8 OK,
   rel <= 1.8e-06 (pure f32 accumulation-order noise; integer dot is bit-exact).
   Covers out 1..128, in 256..2048, partial last warp, chunks=2.
   The old failures were ALL in the sweep harness (kernel matches
   amd.py::_quant_decode_kernel line-for-line). Harness bugs FIXED in place:
   - nibble packing was sequential; must be paired-superblock: qs.reshape(4,2,32),
     low nibble -> sub-block 2k, high -> 2k+1, byte = lo | hi<<4 (== real GGUF layout,
     == amd.py `>> ((subgroup&1)*4)`)
   - scale bytes were mis-packed. Truth (get_scale_min_k4):
     s[j]=sc[j]&63 | (sc[4+j]>>4)<<6; s[4+j]=mn[j]&63 | (mn[4+j]>>4)<<6;
     s[8+j]=sc[4+j]&15 | mn[4+j]<<4
   - reference used signed q-8; real Q4_K weight = d*sc*q - dmin*mn with UNSIGNED q
   - activation quant must be exact: pin each group max to +-127 (scale=1.0), like
     sweep_nv_q8.py did
   - packed weights must be tiled PER OUTPUT ROW: raw = concatenate(blocks * outf);
     kernel indexes base=(output*(in//256)+block)*36
2. **amd.py wired** (committed on branch qwen27b-nv-q8-kernel, fork deven367/tinygrad):
   - import `from tinygrad.llm.kernels.nv_q4k import q4_k_linear` (nv_q4k, NOT nv)
   - `Linear.__call__`: `ggml_type == Q4_K and nv_supported` branch mirroring Q8_0
     (int numel direct, else pad_to(x.max_shape) + shrink)
   - `set_quantized`: packed_sizes now DEVICE-GATED. AMD/RDNA3: QUANT_SIZES dict as
     before. NV: adds Q8_0 (272B) AND Q4_K (144B = Q4_WORDS*4, uint32 view). Other
     devices: nothing claimed (stays decoded -> generic path works).
     This ALSO fixes a pre-existing stock-tinygrad crash: K-quants on NV used to be
     packed then fall through to generic matmul with a flat 1-D weight ->
     transpose IndexError. That is why only the plain Q8_0 27B file ever loaded.
3. **THE "BLOCKING BUG" WAS A BROKEN TEST HARNESS.** `proxy_ab_q4k.py::count_type`
   walked dicts and __dict__ objects but NOT lists — `Transformer.blk` is a plain
   list, so block Linears were never counted; only `.output` was (Q6_K in Q4_K_M
   recipes -> legitimately unclaimed), hence n_q4=0. proxy_ab.py (Q8_0) had
   list handling; the Q4_K copy lost it. Fixed: count_type now walks lists/tuples.
   Detection itself WORKS: qwen3.5:4b Q4_K_M -> 249 Linears: **131 Q4_K claimed,
   48 Q8_0 claimed, 70 unclaimed (Q6_K: output + some ffn weights -> generic, correct)**.
   A/B **32/32 token positions identical** (temp 0), CUSTOM q4_k_linears=131, GENERIC=0.
   Diagnostic: /u/demistry/diag_q4k_graph.py (walk incl. lists, direct set_quantized).
4. **Committed & pushed** (identity deven367 <masterdeven@gmail.com> via
   ~/bin/git-personal):
   - ad94c9619 "add NVIDIA Q4_K custom linear kernel (verified exact: sweep 8/8, proxy 32/32)"
     -> nv_q4k.py (new) + nv.py `_decode_linear` name param
   - 2d46ea489 "route Q4_K on NVIDIA to the custom nv q4_k_linear kernel; gate set_quantized claims by device"
     -> amd.py routing + gated set_quantized
5. **Device gotcha:** tgwork_*.sh export DEV="NVK:$NV+NV" but
   `nv_custom_kernels_supported()` only accepts prefixes NV/CUDA -> "NVK" gates FALSE.
   Device.DEFAULT is already "NV". Run everything WITHOUT setting DEV.

### IN PROGRESS — 27B Q4_K_M benchmark
- File was NOT on node-lair. Downloading
  mradermacher/Qwen3.8-27B-OBLITERATED-GGUF/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf
  (16.8 GB) -> /scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
  (nohup curl PID 3355862, log /tmp/dl_q4km.log).
- CUSTOM: `cd ~/tinygrad-src && python3 -m tinygrad.llm --model
  /scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
  --max_context 512 --benchmark 20`
- GENERIC baseline (same file): `python3 /u/demistry/bench_generic.py --model ...`
  (wrapper sets amd.Linear.use_custom_quant=False then cli main()).
- Compare ONLY vs its own generic path (different model than the Uncensored Q8_0
  file; Q8_0 27B custom-kernel number was 20.9 tok/s). GPU free: check
  `nvidia-smi --query-compute-apps` first.

## Files touched
- node-lair ~/tinygrad-src: tinygrad/llm/kernels/amd.py (M), nv.py (M),
  nv_q4k.py (NEW) — committed ad94c9619..2d46ea489 on qwen27b-nv-q8-kernel, pushed
- node-lair /u/demistry/: sweep_q4k.py (passing), proxy_ab_q4k.py (FIXED list walk,
  passing), diag_q4k_graph.py (new), bench_generic.py (new),
  patch_amd*.py, fix_import.py (scratch, deletable)
- local macOS: /Users/deven367/tmp/{proxy_ab_q4k.py,diag_q4k_graph.py,bench_generic.py,HANDOFF_Q4K.md}
