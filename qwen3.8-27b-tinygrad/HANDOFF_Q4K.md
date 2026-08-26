# Handoff — Q4_K NVIDIA kernel (updated 2026-08-26 late)

## Status: kernel VERIFIED EXACT, wiring landed, ONE bug from proxy A/B green

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
2. **amd.py wired** (all uncommitted on branch qwen27b-nv-q8-kernel, working tree ~/tinygrad-src):
   - import `from tinygrad.llm.kernels.nv_q4k import q4_k_linear` (nv_q4k, NOT nv)
   - `Linear.__call__`: `ggml_type == Q4_K and nv_supported` branch mirroring Q8_0
     (int numel direct, else pad_to(x.max_shape) + shrink)
   - `set_quantized`: packed_sizes now DEVICE-GATED. AMD/RDNA3: QUANT_SIZES dict as
     before. NV: adds Q8_0 (272B) AND Q4_K (144B = Q4_WORDS*4, uint32 view). Other
     devices: nothing claimed (stays decoded -> generic path works).
     This ALSO fixes a pre-existing stock-tinygrad crash: K-quants on NV used to be
     packed then fall through to generic matmul with a flat 1-D weight ->
     transpose IndexError. That is why only the plain Q8_0 27B file ever loaded.
   - regression: sweep_nv_q8.py still 24/24 maxerr=0 after the patch.
3. **Device gotcha:** tgwork_*.sh export DEV="NVK:$NV+NV" but
   `nv_custom_kernels_supported()` only accepts prefixes NV/CUDA -> "NVK" gates FALSE.
   Device.DEFAULT is already "NV". Run everything WITHOUT setting DEV.

### BLOCKING BUG (next step, ~1 debug session)
qwen3.5:4b Q4_K_M proxy A/B (/u/demistry/proxy_ab_q4k.py, run WITHOUT DEV env):
generation runs (generic path, sane text) but after forward ALL Linears have
ggml_type=None -> set_quantized found NO matching `Ops.SHRINK, dtype uint8` node with
prod(shape) in packed_sizes. Detection worked for Q8_0 (shipped kernel proves it).
Suspect: gguf.py loader stores Q4_K (type 12) tensors differently than assumed
(already-bitcast uint32 view? CONTIGUOUS instead of SHRINK? padding offset?), or the
numel//256*144 key mismatches the raw view size. Next command: dump the weight.uop
toposort (uint8/16/32 nodes, dtype+shape+numel) for a mid-model Linear of BOTH
qwen3.5:0.8b (Q8_0, detection works) and qwen3.5:4b (Q4_K_M) and diff the structure.
GOTCHA: set_quantized fires lazily on first Linear.__call__ — counting ggml_type
before generate() always shows None. Only trust post-forward counts.

### After A/B is green (32/32 tokens, n_q4>0)
1. Commit: nv_q4k.py (new), nv.py (_decode_linear name param), amd.py (routing +
   gated set_quantized) on branch qwen27b-nv-q8-kernel in fork deven367/tinygrad.
2. Benchmark 27B: `cd ~/tinygrad-src && python3 -m tinygrad.llm --model
   /scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
   --max_context 512 --benchmark 20`. Compare ONLY vs its own generic path
   (different model than the Uncensored Q8_0 file; rerun generic baseline for it by
   flipping amd.Linear.use_custom_quant=False or temporarily reverting routing).
   GPU is free tonight (llama-server not running; check `nvidia-smi` first anyway).

## Files touched this session
- node-lair ~/tinygrad-src: tinygrad/llm/kernels/amd.py (M), tinygrad/llm/kernels/nv.py (M),
  tinygrad/llm/kernels/nv_q4k.py (new, untracked) — UNCOMMITTED
- node-lair /u/demistry/: sweep_q4k.py (rewritten, passing), proxy_ab_q4k.py (new),
  patch_amd*.py, fix_import.py (applied scratch scripts, deletable)
- local macOS: /Users/deven367/tmp/{patch_amd.py,patch_amd2.py,fix_import.py,proxy_ab_q4k.py}
