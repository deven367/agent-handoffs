#!/usr/bin/env python3
"""Qwen 3.5 4B Q4_K_M greedy-token A/B: NVIDIA custom Q6_K versus generic decode."""
import itertools, sys
from tinygrad import Tensor
from tinygrad.llm.cli import SimpleTokenizer, fetch, models
import tinygrad.llm.kernels.amd as amd

MODEL, PROMPT, N_TOK = "qwen3.5:4b", "The capital of France is", 32


def count_type(obj, ggml_type: int) -> int:
  n = int(isinstance(obj, amd.Linear) and obj.ggml_type == ggml_type)
  try: attrs = vars(obj)
  except TypeError: return n
  for value in attrs.values():
    if isinstance(value, dict): n += sum(count_type(x, ggml_type) for x in value.values())
    elif isinstance(value, (list, tuple)): n += sum(count_type(x, ggml_type) for x in value)
    elif not isinstance(value, (Tensor, int, float, str, type(None))) and hasattr(value, "__dict__"):
      n += count_type(value, ggml_type)
  return n


def run(with_custom: bool, tag: str):
  amd.Linear.use_custom_quant = with_custom
  from tinygrad.llm.model import Transformer
  model, kv = Transformer.from_gguf(fetch(models[MODEL]), 512)
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  tokens = list(itertools.islice(model.generate(tokenizer.encode(PROMPT), temperature=0.0), N_TOK))
  q4_count, q6_count = count_type(model, 12), count_type(model, 14)
  print(f"[{tag}] q4_k_linears={q4_count} q6_k_linears={q6_count} tokens={tokens}")
  print(f"[{tag}] text={tokenizer.decode(tokens)!r}")
  return tokens, q6_count


custom, q6_count = run(True, "CUSTOM")
assert q6_count > 0, "CUSTOM run: no Q6_K Linear engaged"
generic, _ = run(False, "GENERIC")
matched = sum(a == b for a, b in zip(custom, generic))
print(f"MATCH {matched}/{len(custom)} token positions")
sys.exit(0 if matched == len(custom) else 1)
