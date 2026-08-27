#!/usr/bin/env python3
"""Generate deterministic tinygrad token IDs for llama.cpp cross-checking."""
import itertools, json, sys
from tinygrad import Tensor
from tinygrad.llm.cli import SimpleTokenizer
from tinygrad.llm.model import Transformer

MODEL = sys.argv[1]
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "The capital of France is"
TOKEN_COUNT = int(sys.argv[3]) if len(sys.argv) > 3 else 16

Tensor.manual_seed(1234)
model, kv = Transformer.from_gguf(MODEL, 512)
tokenizer = SimpleTokenizer.from_gguf_kv(kv)
prompt_tokens = tokenizer.encode(PROMPT)
tokens = list(itertools.islice(model.generate(prompt_tokens.copy(), temperature=0.0), TOKEN_COUNT))
print(json.dumps({"prompt": PROMPT, "prompt_tokens": prompt_tokens, "tokens": tokens, "text": tokenizer.decode(tokens)}))
