#!/usr/bin/env python3
"""Benchmark tinygrad decode + prefill throughput.

Measures:
  - Decode: tok/s for N autoregressive steps (chunk_size=1)
  - Prefill: tok/s for prompt processing at various chunk sizes

Usage (on node-lair):
  DEV=CUDA python3 qwen3.8-27b-tinygrad/scripts/bench_tinygrad.py [ctx] [decode_steps] [prompt_len]
"""
import sys, time
import tinygrad.llm.model as m
from tinygrad import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.helpers import GlobalCounters, Context, Timing

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"


def bench_decode(ctx: int, steps: int) -> float:
    """Decode tok/s: autoregressive generation from a single seed token."""
    model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")
    # warmup
    model.warmup()
    gen = model.generate([1000], temperature=0.0)
    # burn first token (JIT compiled)
    first = next(gen)
    # timed decode
    GlobalCounters.reset()
    t0 = time.perf_counter()
    for i in range(steps):
        next(gen)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    tok_s = steps / elapsed
    mem_mb = GlobalCounters.mem_used // 1000000
    print(f"decode: {tok_s:.2f} tok/s ({1000/tok_s:.2f} ms/tok) {steps} tokens in {elapsed:.2f}s mem={mem_mb}MB")
    return tok_s


def bench_prefill(ctx: int, prompt_len: int, chunk_sizes: list[int]) -> dict[int, float]:
    """Prefill tok/s for each chunk size. Fresh model per chunk size."""
    results = {}
    for cs in chunk_sizes:
        try:
            model, _ = m.Transformer.from_gguf(MODEL, max_context=ctx, cache_type="f16")
            tokens = [1000 + i for i in range(prompt_len)]
            t0 = time.perf_counter()
            gen = model.generate(tokens, chunk_size=cs, temperature=0.0)
            first = next(gen)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            tok_s = prompt_len / elapsed
            results[cs] = tok_s
            print(f"prefill cs={cs:3d}: {tok_s:8.1f} tok/s ({1000/tok_s:.2f} ms/tok) {prompt_len} tokens in {elapsed:.2f}s first={first}", flush=True)
        except Exception as e:
            print(f"prefill cs={cs:3d}: ERROR: {e}", flush=True)
            results[cs] = 0.0
    return results


def main():
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    decode_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    prompt_len = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    print(f"model={MODEL.split('/')[-1]} ctx={ctx}")
    print()

    # decode benchmark
    print("=== DECODE ===")
    decode_tps = bench_decode(ctx, decode_steps)
    print()

    # prefill benchmark
    print("=== PREFILL ===")
    chunk_sizes = [1, 2, 4, 8, 16, 32]
    prefill_results = bench_prefill(ctx, prompt_len, chunk_sizes)
    print()

    # summary
    print("=== SUMMARY ===")
    print(f"decode: {decode_tps:.2f} tok/s")
    best_cs = max(prefill_results, key=prefill_results.get) if prefill_results else 0
    best_tps = prefill_results.get(best_cs, 0)
    print(f"best prefill: {best_tps:.1f} tok/s at cs={best_cs}")
    print(f"llama.cpp ref: decode=38.83 tok/s, prefill=2611 tok/s")


if __name__ == "__main__":
    main()
