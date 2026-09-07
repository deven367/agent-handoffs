#!/usr/bin/env python3
"""Bisect chunked-prefill divergence: per-block output stats for two chunk sizes.

Loads the model fresh per chunk size (clean recurrent/KV state), hooks
Transformer.forward to stash each block output, then drives the REAL
__call__ path (prefill_jit capture + realize with bound toks/start_pos).
After realize, the stashed intermediates are materialized, so stats are exact.

Usage (on node-lair, from agent-handoffs repo root):
  DEV=CUDA python3 qwen3.8-27b-tinygrad/scripts/bisect_blocks.py "[1]" 1 2 [stop] [tol]
"""
import sys
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"


def block_stats(prompt: list[int], cs: int, stop: int):
  model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
  stashed: list[Tensor] = []
  names: list[str] = []

  orig_forward = model.forward

  def hooked_forward(tokens: Tensor, start_pos, temperature: Tensor):
    x = model.token_embd(tokens).float()
    for i, block in enumerate(model.blk):
      x = block(x, start_pos)
      if i < stop:
        stashed.append(x)
        names.append(type(block).__name__)
    logits = model.output(model.output_norm(x[:, -1:]))[:, -1, :]
    return (logits / temperature.maximum(1e-12)
            - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  model.forward = hooked_forward.__get__(model, type(model))
  try:
    v_start_pos = UOp.variable("start_pos", 0, model.max_context - 1)
    v_toks = UOp.variable("toks", 1, cs)
    t = Tensor(list(prompt) + [0] * (model.max_context - len(prompt)), dtype="int32").reshape(1, model.max_context)
    start_pos = model.get_start_pos(list(prompt))
    n_toks = min(cs, len(prompt) - start_pos)
    sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
    toks = t[:, sp:sp + nt].contiguous()
    temp = Tensor([0.0])
    out = model(toks, sp, temp).realize()
    token = int(out.numpy().reshape(-1)[0])
  finally:
    model.forward = orig_forward
  stats = []
  for y in stashed:
    arr = y.numpy().reshape(-1)
    stats.append((float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())))
  return names, stats, token


def main():
  prompt = eval(sys.argv[1])  # noqa: S307 - local dev script, prompt is a python literal
  cs_a, cs_b = int(sys.argv[2]), int(sys.argv[3])
  stop = int(sys.argv[4]) if len(sys.argv) > 4 else 65
  tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5
  names_a, a, tok_a = block_stats(prompt, cs_a, stop)
  names_b, b, tok_b = block_stats(prompt, cs_b, stop)
  print(f"prompt_len={len(prompt)} cs_a={cs_a}->tok{tok_a} cs_b={cs_b}->tok{tok_b} blocks={len(a)}")
  for i, ((na, sa), (nb, sb)) in enumerate(zip(zip(names_a, a), zip(names_b, b))):
    rel = max(abs(x - y) / max(1e-12, abs(x), abs(y)) for x, y in zip(sa, sb))
    flag = "OK     " if (na == nb and rel <= tol) else "DIVERGE"
    print(f"{flag} blk{i:02d} {na}: a_mean={sa[0]:+.6e} b_mean={sb[0]:+.6e} rel={rel:.2e}", flush=True)
    if flag == "DIVERGE":
      print(f"  a_full={[f'{v:+.6e}' for v in sa]}")
      print(f"  b_full={[f'{v:+.6e}' for v in sb]}")
      break


if __name__ == "__main__":
  main()
