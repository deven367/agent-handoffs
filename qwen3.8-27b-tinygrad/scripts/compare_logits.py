#!/usr/bin/env python3
"""Compare final logits (not argmax) between cs=1 and cs=2 on the same prompt.

Distinguishes float noise (~1e-4) from a logic error (large divergence).
"""
import sys
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.llm.model import Transformer
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
cs = int(sys.argv[1])

PROMPT = [198, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13,
          248046, 198, 248045, 74455, 198, 248068, 271, 248069]

captured = []
orig_forward = Transformer.forward
def hooked(self, tokens, start_pos, temperature):
    x = self.token_embd(tokens).float()
    for block in self.blk: x = block(x, start_pos)
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    captured.append(logits)
    return logits  # raw logits, so the caller can inspect them
Transformer.forward = hooked

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")

# single forward: process the whole prompt in chunks of cs, then read the last logits
v_start_pos = UOp.variable("start_pos", 0, model.max_context - 1)
v_toks = UOp.variable("toks", 1, cs)
t = Tensor(PROMPT + [0]*(model.max_context-len(PROMPT)), dtype="int32").reshape(1, model.max_context)
sp = 0
while sp < len(PROMPT):
    n = min(cs, len(PROMPT) - sp)
    s, nt = v_start_pos.bind(sp), v_toks.bind(n)
    out = model(t[:, s:s+nt], s, Tensor([0.0]))
    sp += n

arr = out.realize().numpy().reshape(-1)
import numpy as np
top = np.argsort(arr)[::-1][:5]
print(f"cs={cs} argmax={int(top[0])}", flush=True)
print(f"cs={cs} top5={[(int(i), round(float(arr[i]), 4)) for i in top]}", flush=True)
print(f"cs={cs} n={arr.size} max={arr.max():+.4f} min={arr.min():+.4f} sum={arr.sum():+.2f}", flush=True)
