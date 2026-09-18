#!/usr/bin/env python3
"""Compare kernel counts: decode (T=1) vs prefill (cs=2). Explains why prefill doesn't amortize."""
import sys, re, collections
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.helpers import GlobalCounters

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")

# --- decode (T=1) ---
gen = model.generate([1, 2, 3, 4], chunk_size=2, temperature=0.0)
next(gen)  # prefill compile
next(gen)  # decode compile
GlobalCounters.reset()
for _ in range(4): next(gen)
n_dec = GlobalCounters.kernel_count / 4
print(f"DECODE (T=1):  {n_dec:.0f} kernels/token", flush=True)

# --- prefill cs=2 ---
prompt = list(range(1000, 1256))
gen = model.generate(prompt, chunk_size=2, temperature=0.0)
next(gen)  # warmup
GlobalCounters.reset()
p2 = list(range(2000, 2256))
gen = model.generate(p2, chunk_size=2, temperature=0.0)
next(gen)
n_pre = GlobalCounters.kernel_count
print(f"PREFILL cs=2:  {n_pre} kernels for 256 tokens = {n_pre/256:.1f} kernels/token", flush=True)
print(f"  ratio vs decode: {n_pre/256/n_dec:.2f}x per token", flush=True)
