#!/usr/bin/env python3
"""Benchmark tinygrad decode throughput.

Usage (on node-lair):
  DEV=CUDA python3 qwen3.8-27b-tinygrad/scripts/bench_decode.py [ctx] [steps]
"""
import sys, time
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters

import os
MODEL = os.environ.get("MODEL")
if not MODEL:
  for candidate in [
    "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf",
    "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf",
  ]:
    if os.path.exists(candidate):
      MODEL = candidate
      break
assert MODEL and os.path.exists(MODEL), f"Model not found: {MODEL}"

ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 512
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20

model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")
gen = model.generate([1000], temperature=0.0)
# burn first 5 tokens (JIT compile prefill + rollout + graph capture)
for _ in range(5):
    next(gen)
# timed decode
GlobalCounters.reset()
t0 = time.perf_counter()
for i in range(steps):
    next(gen)
t1 = time.perf_counter()
elapsed = t1 - t0
tok_s = steps / elapsed
mem_mb = GlobalCounters.mem_used // 1000000
print(f"decode: {tok_s:.2f} tok/s ({1000/tok_s:.2f} ms/tok) {steps} tokens in {elapsed:.2f}s mem={mem_mb}MB")
