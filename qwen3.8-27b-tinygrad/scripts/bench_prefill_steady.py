#!/usr/bin/env python3
"""Final prefill benchmark: 3 warmup runs, then 3 timed runs (steady state)."""
import sys, time, os
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
NTOK = 256

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")

# use distinct prompts each run so start_pos never satisfies the reuse condition (start_pos stays 0)
def mkprompt(seed): return [((seed*7919 + i*104729) % 100000) + 1 for i in range(NTOK)]

def run(tag):
    p = mkprompt(len(model._cached_tokens) + 1000)  # always-divergent prompt => full prefill
    t0 = time.perf_counter()
    gen = model.generate(p, chunk_size=2, temperature=0.0)
    next(gen)
    t1 = time.perf_counter()
    print(f"  {tag}: {NTOK/(t1-t0):7.1f} tok/s  ({(t1-t0)*1000:7.1f} ms)", flush=True)

for i in range(3): run(f"warmup {i}")
print("--- steady state ---")
for i in range(3): run(f"timed  {i}")
