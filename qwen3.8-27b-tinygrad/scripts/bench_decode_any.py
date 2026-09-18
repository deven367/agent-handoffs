#!/usr/bin/env python3
"""Quick decode benchmark — works on any GPU."""
import sys, time
import tinygrad.llm.model as m

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 512
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20

model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")
gen = model.generate([1000], temperature=0.0)
next(gen); next(gen)  # JIT compile

t0 = time.perf_counter()
for _ in range(steps):
    next(gen)
t1 = time.perf_counter()
print(f"decode: {steps/(t1-t0):.2f} tok/s ({(t1-t0)/steps*1000:.2f} ms/tok) {steps} tokens in {t1-t0:.2f}s")
