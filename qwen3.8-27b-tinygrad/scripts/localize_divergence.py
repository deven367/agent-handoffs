#!/usr/bin/env python3
"""Localize T=1 vs T=4 divergence: embedding output vs first SSM block, token 0 only."""
import sys
import numpy as np
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.llm.model import Transformer
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
PROMPT = [198, 248045, 846, 198]

def run(cs, ntok):
    st = {}
    orig = Transformer.forward
    def hooked(self, tokens, start_pos, temperature):
        x = self.token_embd(tokens).float()
        st['emb'] = x
        x = self.blk[0](x, start_pos)
        st['blk0'] = x
        for block in self.blk[1:]: x = block(x, start_pos)
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
    out = {}
    for k, v in st.items():
        out[k] = v.shrink(((0,1),(0,1),(0,v.shape[2]))).realize().numpy().reshape(-1)
    return out

a = run(int(sys.argv[1]), int(sys.argv[2]))
b = run(int(sys.argv[3]), int(sys.argv[4]))
for k in ("emb", "blk0"):
    x, y = a[k], b[k]
    scale = max(1e-9, float(np.abs(x).max()))
    mx = float(np.abs(x - y).max())
    print(f"{k:5s} max|x|={scale:.6e} max|d|={mx:.6e} rel={mx/scale:.2e}", flush=True)
print(f"  emb[:6]  cs1={[round(float(v),5) for v in a['emb'][:6]]}")
print(f"  emb[:6]  cs4={[round(float(v),5) for v in b['emb'][:6]]}")
