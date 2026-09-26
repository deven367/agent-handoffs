#!/usr/bin/env python3
"""Greedy token A/B + decode rate for the 27B Q4_K_M model.

`make parity` only validates logits — it does NOT catch a broken sampler. Run this
before and after any change to `Transformer.forward`'s sampling tail and require the
TOKENS line to be byte-identical.

  DEV=CUDA PYTHONPATH=$HOME/projects/tinygrad-src \
  MODEL=/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf \
  python3 qwen3.8-27b-tinygrad/scripts/greedy_token_ab.py

Baseline at tinygrad 38342a3be (H100, 2026-09-25):
  TOKENS [381, 310, 5790, 421, 279, 491, 2936, 1000, 381]
  decode: 69.78 tok/s (14.33 ms/tok)
The rate is A/B-only (timing starts at token 9, so it reads lower than `make bench-tg`);
never quote it as the headline number.
"""
import os
import time

import tinygrad.llm.model as m
from tinygrad.helpers import GlobalCounters

MODEL = os.environ.get("MODEL", "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf")
TOKENS = 9
STEPS = 20

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
gen = model.generate([1000], temperature=0.0)
ids = [next(gen) for _ in range(TOKENS)]
print("TOKENS", ids, flush=True)
GlobalCounters.reset()
t0 = time.perf_counter()
for _ in range(STEPS):
  next(gen)
dt = time.perf_counter() - t0
print(f"decode: {STEPS/dt:.2f} tok/s ({1000*dt/STEPS:.2f} ms/tok)", flush=True)
