#!/usr/bin/env python3
"""Profile a single decode step and print per-kernel timings.

Usage (on node-lair):
  DEV=CUDA DEBUG=2 python3 qwen3.8-27b-tinygrad/scripts/profile_decode.py
"""
import sys, time, re, collections
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.helpers import GlobalCounters

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
gen = model.generate([1000], temperature=0.0)
next(gen)  # prefill JIT compile
next(gen)  # rollout JIT compile

# profile one decode step
GlobalCounters.reset()
t0 = time.perf_counter()
next(gen)
t1 = time.perf_counter()
print(f"\n=== DECODE STEP: {1e3*(t1-t0):.2f} ms, {GlobalCounters.global_mem/1e9:.2f} GB, {GlobalCounters.global_ops/1e9:.2f} GFLOPS ===", flush=True)
print(f"kernels: {GlobalCounters.kernel_count}", flush=True)
print(f"mem used: {GlobalCounters.mem_used//1000000} MB", flush=True)
