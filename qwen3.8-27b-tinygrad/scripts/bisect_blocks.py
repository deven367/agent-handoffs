#!/usr/bin/env python3
"""Bisect chunked-prefill divergence: per-block output stats for two chunk sizes.

Loads the model fresh for each chunk size (clean recurrent/KV state), replays
the exact first-chunk setup from Transformer.generate(), walks the blocks,
realizes each block output (handles symbolic T via bound var_vals), and
reports the first diverging block.

Usage (on node-lair, after `git pull` in agent-handoffs):
  DEV=CUDA python3 scripts/bisect_blocks.py "[1]" 1 2 [stop] [tol]
"""
import sys
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"


def block_stats(prompt: list[int], cs: int, stop: int):
  model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
  v_start_pos = UOp.variable("start_pos", 0, model.max_context - 1)
  v_toks = UOp.variable("toks", 1, cs)
  t = Tensor(list(prompt) + [0] * (model.max_context - len(prompt)), dtype="int32").reshape(1, model.max_context)
  start_pos = model.get_start_pos(list(prompt))
  n_toks = min(cs, len(prompt) - start_pos)
  sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
  x = model.token_embd(t[:, sp:sp + nt].contiguous()).float()
  stats = []
  for i, block in enumerate(model.blk):
    if i >= stop:
      break
    y = block(x, sp).realize()
    arr = y.numpy().reshape(-1)
    stats.append((type(block).__name__, float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())))
    x = y
  return stats


def main():
  prompt = eval(sys.argv[1])  # noqa: S307 - local dev script, prompt is a python literal
  cs_a, cs_b = int(sys.argv[2]), int(sys.argv[3])
  stop = int(sys.argv[4]) if len(sys.argv) > 4 else 65
  tol = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5
  a = block_stats(prompt, cs_a, stop)
  b = block_stats(prompt, cs_b, stop)
  print(f"prompt_len={len(prompt)} cs_a={cs_a} cs_b={cs_b} blocks={len(a)}")
  for i, ((na, *sa), (nb, *sb)) in enumerate(zip(a, b)):
    rel = max(abs(x - y) / max(1e-12, abs(x), abs(y)) for x, y in zip(sa, sb))
    flag = "OK " if (na == nb and rel <= tol) else "DIVERGE"
    print(f"{flag} blk{i:02d} {na}: a_mean={sa[0]:+.6e} b_mean={sb[0]:+.6e} rel={rel:.2e}")
    if flag == "DIVERGE":
      print(f"  a_full={[f'{v:+.6e}' for v in sa]}")
      print(f"  b_full={[f'{v:+.6e}' for v in sb]}")
      break


if __name__ == "__main__":
  main()
