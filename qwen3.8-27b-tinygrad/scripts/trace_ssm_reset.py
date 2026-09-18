#!/usr/bin/env python3
"""Instrument gated_delta_prefill: which branch (reset vs keep) is taken per call?"""
import sys
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.llm.kernels import amd as K
from tinygrad.uop.ops import resolve

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"

orig = K.gated_delta_prefill
calls = []
def traced(q, k, v, beta, alpha, state, start_pos=None):
    if start_pos is None:
        tag = "start_pos=None -> ALWAYS RESET"
    else:
        _, sp_val = start_pos.uop.unbind()
        is0 = bool(resolve(sp_val == 0))
        tag = f"sp_val={sp_val} -> {'RESET' if is0 else 'KEEP'}"
    calls.append(tag)
    return orig(q, k, v, beta, alpha, state, start_pos)
K.gated_delta_prefill = traced
m.gated_delta_prefill = traced

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")

prompt = [198, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069]
print(f"prompt len = {len(prompt)}")

for cs in (1, 2):
    calls.clear()
    g = model.generate(prompt, chunk_size=cs, temperature=0.0)
    out = [next(g) for _ in range(4)]
    print(f"\n=== cs={cs} -> first 4 tokens {out} ===")
    print(f"  gated_delta_prefill traced {len(calls)} times; unique: {sorted(set(calls))}")
    for i, c in enumerate(calls[:8]):
        print(f"    [{i}] {c}")
