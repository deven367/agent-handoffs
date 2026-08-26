# Handoff — Q4_K NVIDIA kernel (2026-08-26)

## Context
Task: optimize Qwen3.8-27B inference through tinygrad beyond the shipped Q8_0 kernel.
Q8_0 kernel shipped: **2.06 → 20.9 tok/s** (9.4× from baseline, +7.6% from `__ldcs`
streaming loads). Verified: 24/24 + 4/4 sweeps maxerr=0, proxy A/B 32/32 tokens, 27B
benchmark on L40S. Branch `qwen27b-nv-q8-kernel` in fork `deven367/tinygrad` (4 commits
on d851aca9a).

## Current work — Q4_K decode kernel (IN PROGRESS)

**Goal**: 4-bit weights halve memory traffic → decode ~30-40 t/s potential.

### Status
- `tinygrad/llm/kernels/nv_q4k.py` written (Q4_K 256-weight block, mirrors amd.py
  `_quant_decode_kernel`): `dot·d·scale − qsum·dmin·minimum`, 32-element groups, warp
  shuffle reduce.
- `nv.py` `_decode_linear` refactored to take a `name` param (default `nv_linear_q8_0`)
  so Q4_K reuses the exact same decoder tail.
- **Blocking bug in verification**: sweep outputs are WRONG (maxerr huge) but the kernel
  now RUNS. The last fix (reference `q-8` → unsigned `q`) is **made but NOT yet run** —
  run the sweep to see if the failure is the sweep's numpy reference or the kernel.
  - Sweep: `/u/demistry/sweep_q4k.py` (random blocks vs numpy exact).
  - If still failing: suspect the kernel's `scale`/`minimum` decode or the `qsum` path.
    Compare against amd.py `_q5_scales` + `_quant_decode_kernel` (lines ~107-170) which
    is proven-exact. Also check `d = _half(raw[base] & 0xffff)` — note `_half` takes a
    uint16; `raw[base] & 0xffff` on a uint32 may truncate via cast — verify the rendered
    source (`/u/demistry/dump_dec_src.py` pattern) if values are still off.

### After sweep passes
1. Wire into `amd.py::Linear.__call__`: route `ggml_type == 12` (Q4_K) on NV/CUDA to
   `nv_q4k.q4_k_linear`. Mirror the Q8_0 pattern: `set_quantized` needs a Q4_K entry in
   `packed_sizes` for NV — **144 bytes per 256-weight block** (already in QUANT_SIZES for
   AMD; the NV-only dict build adds Q8_0; add Q4_K there too, with `nv_custom_kernels_supported`
   gating, NOT the uint16 view — Q4_K blocks are 144=36×uint32, use uint32 view).
2. Proxy A/B: qwen3.5:0.8b is Q8_0 — for Q4_K use `qwen3.5:4b` (Q4_K_M per cli.py) — 
   greedy tokens CUSTOM vs GENERIC must match (small tolerance — activation quant lossy).
3. Benchmark: `Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` (16.8 GB, downloaded by user). NOTE:
   this is a DIFFERENT model than Uncensored Q8_0 — compare only vs its own generic path.

## Critical gotchas (learned this session)
1. **Unaligned memory access on this stack is BROKEN**: `*(uint*)((char*)ptr+2)` and
   misaligned `__ldcs` → kernel hang (signal 12 oom/never-completes) or garbage.
   Q8_0's 34-byte block is 2-aligned; Q4_K's 144-byte block is 4-aligned — **use only
   aligned uint32 element loads**.
2. Slurm box: ssh-adopted sessions into the user's interactive job (84940-era) got
   EPERM on `/dev/nvidia0` — fresh allocations (`sbatch` from login node `lair`) had
   working GPUs. Current state: USER fixed access (job 84953) — direct ssh to node-lair
   has GPU now; verify with `python3 -c "import os;os.open('/dev/nvidia0',os.O_RDWR)"`.
3. `/tmp` on the node gets wiped; keep artifacts in `~/tinygrad-src` / `$HOME` logs
   (`/u/demistry/*.log`). Sweep scripts live at `/u/demistry/sweep_*.py`.
4. tinygrad kernels compile in WORKER subprocesses — DEBUG asm from the driver process
   is empty. Dump kernel source via `to_program(k, Device["NV"].renderer)` and read
   `u.op is Ops.SOURCE` arg (pattern in `/u/demistry/dump_dec_src.py`).

## Repos
- tinygrad fork: `deven367/tinygrad`, branch `qwen27b-nv-q8-kernel` (working tree
  `~/tinygrad-src` on node-lair; local mirror `/Users/deven367/tmp/tinygrad-work`).
- Handoff doc: `/u/demistry/agent-handoffs/qwen3.8-27b-tinygrad/progress.md` (git repo
  `deven367/agent-handoffs`, main).