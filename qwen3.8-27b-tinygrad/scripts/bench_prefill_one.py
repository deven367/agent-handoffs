#!/usr/bin/env python3
"""Benchmark tinygrad prefill throughput for a single chunk size.

One process per chunk size (caller runs multiple). Does a warmup prefill pass
first to JIT-compile, then times the real prefill.

Usage (on node-lair):
  DEV=CUDA python3 qwen3.8-27b-tinygrad/scripts/bench_prefill_one.py [ctx] [prompt_len] [cs]
"""
import sys, time
import tinygrad.llm.model as m
from tinygrad import Tensor

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"

ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 512
prompt_len = int(sys.argv[2]) if len(sys.argv) > 2 else 256
cs = int(sys.argv[3]) if len(sys.argv) > 3 else 32

model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")

# warmup: short prefill to JIT-compile the graph
warmup_tokens = [1000 + i for i in range(min(cs, 8))]
list(model.generate(warmup_tokens, chunk_size=cs, temperature=0.0))

# real prefill: time prompt processing
tokens = [1000 + i for i in range(prompt_len)]
t0 = time.perf_counter()
gen = model.generate(tokens, chunk_size=cs, temperature=0.0)
first = next(gen)
t1 = time.perf_counter()

elapsed = t1 - t0
tok_s = prompt_len / elapsed
print(f"cs={cs:3d}  {tok_s:8.1f} tok/s  ({1000/tok_s:.2f} ms/tok)  {prompt_len} tokens in {elapsed:.2f}s  first={first}")
