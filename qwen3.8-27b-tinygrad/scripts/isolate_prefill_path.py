#!/usr/bin/env python3
"""Isolate prefill path: same prompt, cs=1 (token-by-token) vs cs=2 (chunked)."""
import sys, json
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import jinja2
import tinygrad.llm.model as m
from tinygrad.llm.cli import SimpleTokenizer

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
model, kv = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
tok = SimpleTokenizer.from_gguf_kv(kv)

ct = kv.get("tokenizer.chat_template")
env = jinja2.Environment()
env.filters['tojson'] = lambda obj, **kw: json.dumps(obj, **kw)
env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
env.globals['strftime_now'] = lambda fmt: "x"
env.globals['bos_token'] = tok.decode([tok.bos_id]) if tok.bos_id is not None else ""
env.globals['eos_token'] = tok.decode([tok.eos_id])
tmpl = env.from_string(ct)

msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
text = tmpl.render(messages=msgs, add_generation_prompt=True, preserve_thinking=True)
ids = tok.encode(text)
print(f"prompt tokens: {len(ids)}")

for cs in (1, 2):
    g = model.generate(ids, chunk_size=cs, temperature=0.0)
    out = [next(g) for _ in range(16)]
    print(f"cs={cs}: tokens={out}")
    print(f"cs={cs}: text={tok.decode(out)!r}")
