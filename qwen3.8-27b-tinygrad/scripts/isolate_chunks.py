#!/usr/bin/env python3
"""Isolate: does a SINGLE prefill chunk (cs=len) match cs=1, or is all T>1 broken?"""
import sys
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.llm.cli import SimpleTokenizer

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
cs = int(sys.argv[1])

model, kv = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
tok = SimpleTokenizer.from_gguf_kv(kv)

# fixed 24-token prompt (rendered Qwen chat template)
prompt = [198, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13,
          248046, 198, 248045, 74455, 198, 248068, 271, 248069]
g = model.generate(prompt, chunk_size=cs, temperature=0.0)
out = [next(g) for _ in range(4)]
print(f"cs={cs:2d} chunks={-(-len(prompt)//cs):2d} first4={out} text={tok.decode(out)!r}", flush=True)
