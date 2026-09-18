#!/usr/bin/env python3
"""Fresh process per chunk size: trace SSM reset/keep and report generated tokens."""
import sys
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.llm.kernels import amd as K
from tinygrad.uop.ops import resolve

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
cs = int(sys.argv[1])

calls = []
orig = K.gated_delta_prefill
def traced(q, k, v, beta, alpha, state, start_pos=None):
    if start_pos is None: tag = "None->RESET"
    else:
        _, sp = start_pos.uop.unbind()
        tag = f"sp={sp}->{'RESET' if bool(resolve(sp == 0)) else 'KEEP'}"
    calls.append(tag)
    return orig(q, k, v, beta, alpha, state, start_pos)
K.gated_delta_prefill = traced
m.gated_delta_prefill = traced

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
prompt = [198, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069]
print(f"cs={cs} start_pos={model.get_start_pos(prompt)} prompt_len={len(prompt)}", flush=True)

g = model.generate(prompt, chunk_size=cs, temperature=0.0)
out = [next(g) for _ in range(4)]
print(f"cs={cs} first4={out}", flush=True)
print(f"cs={cs} traces={len(calls)} unique={sorted(set(calls))}", flush=True)
# show the sequence of decisions taken during tracing
from collections import Counter
print(f"cs={cs} counts={dict(Counter(calls))}", flush=True)
