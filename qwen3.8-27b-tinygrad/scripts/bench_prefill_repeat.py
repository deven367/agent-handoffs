#!/usr/bin/env python3
"""Prefill benchmark with repeated runs to expose warmup effects (cs=2, 256 tokens)."""
import sys, time, os
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.helpers import GlobalCounters

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
NTOK, REPS = 256, 4

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
prompt = list(range(1, NTOK+1))

def run(tag):
    t0 = time.perf_counter()
    gen = model.generate(prompt, chunk_size=2, temperature=0.0)
    tok = next(gen)
    t1 = time.perf_counter()
    print(f"  {tag}: {NTOK/(t1-t0):7.1f} tok/s  ({(t1-t0)*1000:7.1f} ms total, tok={tok})", flush=True)

for i in range(REPS):
    run(f"run {i} (start_pos={model.get_start_pos(prompt)})")
