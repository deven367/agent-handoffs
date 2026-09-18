#!/usr/bin/env python3
"""Decisive: is a SINGLE T>1 chunk correct, or is all T>1 broken?

Compares logits for the same 4-token prompt processed as:
  cs=1 -> 4 chunks of 1 token
  cs=4 -> 1 chunk of 4 tokens
If these agree, multi-chunk state carry-over is the bug. If they differ, T>1 itself is broken.
"""
import sys
import numpy as np
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
cs = int(sys.argv[1])
PROMPT = [198, 248045, 846, 198]  # 4 tokens

from tinygrad.llm.model import Transformer
def hooked(self, tokens, start_pos, temperature):
    x = self.token_embd(tokens).float()
    for block in self.blk: x = block(x, start_pos)
    return self.output(self.output_norm(x[:, -1:]))[:, -1, :]  # raw logits
Transformer.forward = hooked

model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
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
top = np.argsort(arr)[::-1][:3]
print(f"cs={cs} argmax={int(top[0])} top3={[(int(i), round(float(arr[i]),3)) for i in top]} sum={arr.sum():+.1f}", flush=True)
