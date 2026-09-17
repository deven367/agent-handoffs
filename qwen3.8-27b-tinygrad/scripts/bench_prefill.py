#!/usr/bin/env python3
"""Benchmark tinygrad prefill throughput at various chunk sizes.

Fresh process per chunk size. Each run loads model, times prompt processing,
prints tok/s. No warmup (first call JIT-compiles, included in timing — acceptable
for prefill since the prefill graph is what we're measuring).

Usage (on node-lair):
  DEV=CUDA python3 qwen3.8-27b-tinygrad/scripts/bench_prefill.py [ctx] [prompt_len] [cs1 cs2 ...]
"""
import sys, time
import tinygrad.llm.model as m
from tinygrad import Tensor

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"

ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 512
prompt_len = int(sys.argv[2]) if len(sys.argv) > 2 else 256
if len(sys.argv) > 3:
    chunk_sizes = [int(x) for x in sys.argv[3:]]
else:
    chunk_sizes = [1, 2, 4, 8, 16, 32]

print(f"model=Qwen3.8-27B-OBLITERATED-Q4_K_M ctx={ctx} prompt_len={prompt_len}")
print(f"{'cs':>4s}  {'tok/s':>8s}  {'ms/tok':>8s}  {'first':>6s}")
print("-" * 40)

for cs in chunk_sizes:
    try:
        model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")
        tokens = [1000 + i for i in range(prompt_len)]
        t0 = time.perf_counter()
        gen = model.generate(tokens, chunk_size=cs, temperature=0.0)
        first = next(gen)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        tok_s = prompt_len / elapsed
        print(f"{cs:4d}  {tok_s:8.1f}  {1000/tok_s:8.2f}  {first:6d}", flush=True)
    except Exception as e:
        print(f"{cs:4d}  ERROR: {e}", flush=True)
