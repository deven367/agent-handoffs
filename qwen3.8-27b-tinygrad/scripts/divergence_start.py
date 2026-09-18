#!/usr/bin/env python3
"""Where does T=1 vs T=4 divergence start? Compare TOKEN 0's block output only.

Both runs process the same first token at start_pos=0, so token 0's block outputs
must match. Compares the first `nblk` blocks and prints the first one that differs.
"""
import sys
import numpy as np
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.llm.model import Transformer
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
PROMPT = [198, 248045, 846, 198]
NBLK = int(sys.argv[1]) if len(sys.argv) > 1 else 4

def run(cs, ntok):
    stashed = []
    orig = Transformer.forward
    def hooked(self, tokens, start_pos, temperature):
        x = self.token_embd(tokens).float()
        for i, block in enumerate(self.blk):
            x = block(x, start_pos)
            if i < NBLK: stashed.append((i, x))
        return self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    Transformer.forward = hooked
    try:
        model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
        vsp = UOp.variable("start_pos", 0, model.max_context - 1)
        vt = UOp.variable("toks", 1, cs)
        t = Tensor(PROMPT + [0]*(model.max_context-len(PROMPT)), dtype="int32").reshape(1, model.max_context)
        s, nt = vsp.bind(0), vt.bind(ntok)
        model(t[:, s:s+nt], s, Tensor([0.0])).realize()
    finally:
        Transformer.forward = orig
    # token 0 only
    return {i: y.shrink(((0,1),(0,1),(0,y.shape[2]))).realize().numpy().reshape(-1)
            for i, y in stashed[:NBLK]}

a = run(1, 1)    # T=1, token 0 alone
b = run(4, 4)    # T=4, tokens 0..3
print(f"comparing token 0 across {len(a)} blocks (cs=1 vs cs=4)", flush=True)
for i in sorted(a):
    x, y = a[i], b[i]
    scale = max(1e-9, float(np.abs(x).max()))
    reld = float(np.abs(x - y).max()) / scale
    flag = "OK     " if reld < 1e-3 else "DIVERGE"
    print(f"{flag} blk{i:02d} max|x|={scale:.6e} max|d|={np.abs(x-y).max():.6e} rel={reld:.2e}", flush=True)
    if flag == "DIVERGE": break
