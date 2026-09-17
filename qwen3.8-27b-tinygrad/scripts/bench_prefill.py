#!/usr/bin/env python3
"""Benchmark prefill throughput at various chunk sizes.

Loads the model once, then times prompt processing for each chunk size.
Fresh process per chunk size to avoid JIT cache contamination.

Usage (on node-lair):
  DEV=CUDA python3 qwen3.8-27b-tinygrad/scripts/bench_prefill.py [ctx] [prompt_len]
"""
import sys, time
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.uop.ops import UOp

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"


def bench_chunk_size(cs: int, prompt_len: int, ctx: int) -> tuple[float, int]:
    """Returns (prefill_tok_s, first_token)."""
    model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")
    # build a dummy prompt
    tokens = [1000 + i for i in range(prompt_len)]
    t0 = time.perf_counter()
    gen = model.generate(tokens, chunk_size=cs, temperature=0.0)
    first_token = next(gen)
    t1 = time.perf_counter()
    # prefill time = time to process all prompt tokens and produce first token
    # subtract approximate decode time for the 1 generated token
    prefill_time = t1 - t0
    tok_s = prompt_len / prefill_time
    return tok_s, first_token


def main():
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    prompt_len = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    chunk_sizes = [1, 2, 4, 8, 16, 32]
    if ctx < 512:
        chunk_sizes = [cs for cs in chunk_sizes if cs <= 32]

    print(f"model={MODEL.split('/')[-1]} ctx={ctx} prompt_len={prompt_len}")
    print(f"{'cs':>4s}  {'tok/s':>8s}  {'ms/tok':>8s}  {'first':>6s}")
    print("-" * 40)

    for cs in chunk_sizes:
        try:
            tok_s, first = bench_chunk_size(cs, prompt_len, ctx)
            print(f"{cs:4d}  {tok_s:8.1f}  {1000/tok_s:8.2f}  {first:6d}", flush=True)
        except Exception as e:
            print(f"{cs:4d}  ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
