#!/usr/bin/env python3
"""Sweep chunk sizes with proper warmup, reporting memory and speed."""
import sys, time, os
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.helpers import GlobalCounters

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
mode = "FP16" if os.getenv("PREFILL_FP16", "0") != "0" else "GEMV"
NTOK = 256

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
print(f"[{mode}] model loaded, mem={GlobalCounters.mem_used//1000000} MB", flush=True)

for cs in [int(x) for x in (sys.argv[1:] or ["2", "4", "8", "16", "32"])]:
    try:
        # full warmup: compile prefill at this cs AND decode
        gen = model.generate(list(range(1, NTOK+1)), chunk_size=cs, temperature=0.0)
        next(gen)
        # timed run
        t0 = time.perf_counter()
        gen = model.generate(list(range(1, NTOK+1)), chunk_size=cs, temperature=0.0)
        next(gen)
        t1 = time.perf_counter()
        print(f"[{mode}] cs={cs:3d}  {NTOK/(t1-t0):7.1f} tok/s  ({(t1-t0)*1000/NTOK:6.2f} ms/tok)  mem={GlobalCounters.mem_used//1000000} MB", flush=True)
    except Exception as e:
        print(f"[{mode}] cs={cs:3d}  FAILED: {type(e).__name__}: {str(e)[:80]}", flush=True)
