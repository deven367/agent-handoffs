#!/usr/bin/env python3
"""Prefill benchmark: fresh process per chunk size, full warmup at target cs.

Usage: python bench_prefill_final.py <cs> [ntokens]
"""
import sys, time, os
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.helpers import GlobalCounters

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
cs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 256
mode = "FP16" if os.getenv("PREFILL_FP16", "0") != "0" else "GEMV"

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
print(f"[{mode}] loaded mem={GlobalCounters.mem_used//1000000} MB", flush=True)

prompt = list(range(1, NTOK+1))
try:
    gen = model.generate(prompt, chunk_size=cs, temperature=0.0)
    next(gen)  # warmup at target cs (same shape)
    print(f"[{mode}] warmup done mem={GlobalCounters.mem_used//1000000} MB", flush=True)
    t0 = time.perf_counter()
    gen = model.generate(prompt, chunk_size=cs, temperature=0.0)
    next(gen)
    t1 = time.perf_counter()
    print(f"[{mode}] cs={cs:3d}  {NTOK/(t1-t0):7.1f} tok/s  ({(t1-t0)*1000/NTOK:6.2f} ms/tok)  mem={GlobalCounters.mem_used//1000000} MB", flush=True)
except Exception as e:
    print(f"[{mode}] cs={cs:3d}  FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
