#!/usr/bin/env python3
"""Profile a single prefill step on H100 to understand the slowdown."""
import sys, time
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters, DEBUG
from tinygrad.uop.ops import UOp

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
ctx = 512
cs = 2

model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")

# Warmup
gen = model.generate([1, 2, 3, 4], chunk_size=cs, temperature=0.0)
next(gen); next(gen)

# Profile a single prefill step
GlobalCounters.reset()
prompt = list(range(1, 257))
t0 = time.perf_counter()
gen = model.generate(prompt, chunk_size=cs, temperature=0.0)
first = next(gen)
t1 = time.perf_counter()
print(f"prefill cs={cs}: {(t1-t0)*1000:.1f} ms for {len(prompt)} tokens ({len(prompt)/(t1-t0):.1f} tok/s)")
print(f"kernels: {GlobalCounters.kernel_count}")
print(f"global_ops: {GlobalCounters.global_ops}")
print(f"global_mem: {GlobalCounters.global_mem}")
print(f"mem_used: {GlobalCounters.mem_used//1000000} MB")
