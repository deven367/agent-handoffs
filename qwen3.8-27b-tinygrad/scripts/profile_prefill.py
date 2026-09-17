#!/usr/bin/env python3
"""Profile a cs=2 prefill step to find the bottleneck."""
import time
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters, DEBUG
import os

os.environ["DEBUG"] = "4"

model, _ = m.Transformer.from_gguf(
    "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf",
    max_context=512, cache_type="f16")

# warmup with cs=2
list(model.generate([1000, 1001, 1002, 1003], chunk_size=2, temperature=0.0))

# profile one cs=2 step
GlobalCounters.reset()
t0 = time.perf_counter()
gen = model.generate([1000+i for i in range(256)], chunk_size=2, temperature=0.0)
first = next(gen)
t1 = time.perf_counter()
print(f"\n=== PREFILL cs=2: {1e3*(t1-t0):.1f} ms for 256 tokens ({256/(t1-t0):.1f} tok/s) ===", flush=True)
print(f"kernels: {GlobalCounters.kernel_count}", flush=True)
print(f"mem used: {GlobalCounters.mem_used//1000000} MB", flush=True)
