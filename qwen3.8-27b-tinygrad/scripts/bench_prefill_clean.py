#!/usr/bin/env python3
"""Clean prefill benchmark: minimal warmup, proper timing."""
import sys, time
import tinygrad.llm.model as m
from tinygrad import Tensor

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 512
prompt_len = int(sys.argv[2]) if len(sys.argv) > 2 else 256
cs = int(sys.argv[3]) if len(sys.argv) > 3 else 2

model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")

# Minimal warmup: just 2 prefill steps + 1 rollout to JIT-compile
# Don't generate hundreds of tokens (fills cache, wastes time)
warmup_gen = model.generate([1000, 1001, 1002, 1003], chunk_size=cs, temperature=0.0)
next(warmup_gen)  # prefill JIT compile (2 steps of cs=2)
next(warmup_gen)  # rollout JIT compile (1 step)
warmup_gen.close()  # stop generating

# Reset model state for real prefill
model._cached_tokens = []

# Real prefill: time prompt processing
tokens = [1000 + i for i in range(prompt_len)]
t0 = time.perf_counter()
gen = model.generate(tokens, chunk_size=cs, temperature=0.0)
first = next(gen)
t1 = time.perf_counter()

elapsed = t1 - t0
tok_s = prompt_len / elapsed
print(f"cs={cs:3d}  {tok_s:8.1f} tok/s  ({1000/tok_s:.2f} ms/tok)  {prompt_len} tokens in {elapsed:.2f}s  first={first}")
