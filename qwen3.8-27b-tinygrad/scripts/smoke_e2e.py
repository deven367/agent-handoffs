#!/usr/bin/env python3
"""End-to-end smoke test on committed code: prefill + decode produce coherent text."""
import sys, time
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
import tinygrad.llm.model as m
from tinygrad.llm.cli import SimpleTokenizer

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"

prompt_text = "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"

model, kv = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
tok = SimpleTokenizer.from_gguf_kv(kv)

tokens = tok.encode(prompt_text)
print(f"prompt: {prompt_text!r}")
print(f"prompt tokens ({len(tokens)}): {tokens}")

t0 = time.perf_counter()
gen = model.generate(tokens, chunk_size=2, temperature=0.0)
out = [next(gen) for _ in range(16)]
t1 = time.perf_counter()
print(f"\ngenerated {len(out)} tokens in {t1-t0:.2f}s")
print(f"token ids: {out}")
print(f"text: {tok.decode(out)!r}")
