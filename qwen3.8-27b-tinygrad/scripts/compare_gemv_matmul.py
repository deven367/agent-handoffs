#!/usr/bin/env python3
"""Compare GEMV vs f16 matmul for multi-token input, using TinyJit."""
import time
from tinygrad import Tensor, dtypes, TinyJit
import tinygrad.llm.model as m

model, _ = m.Transformer.from_gguf(
    "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf",
    max_context=512, cache_type="f16")

blk = model.blk[0]
linear = blk.ssm_qkv if hasattr(blk, "ssm_qkv") else blk.ssm_out
print(f"layer: in={linear.in_features} out={linear.out_features} type={linear.ggml_type}")

w = Tensor.randn(linear.in_features, linear.out_features, dtype=dtypes.float16).realize()

for toks in [1, 2, 4, 8, 16, 32]:
    x = Tensor.randn(1, toks, linear.in_features, dtype=dtypes.float16).realize()

    @TinyJit
    def jit_gemv(x):
        return linear(x).realize()

    @TinyJit
    def jit_matmul(x):
        return x.linear(w).realize()

    # warmup (JIT compile)
    jit_gemv(x); jit_matmul(x)
    jit_gemv(x); jit_matmul(x)

    # timed
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        jit_gemv(x)
    t1 = time.perf_counter()
    gemv_ms = (t1 - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        jit_matmul(x)
    t1 = time.perf_counter()
    matmul_ms = (t1 - t0) / N * 1000

    print(f"toks={toks:3d}  GEMV={gemv_ms:7.3f}ms ({gemv_ms/toks:6.3f}ms/tok)  "
          f"matmul={matmul_ms:7.3f}ms ({matmul_ms/toks:6.3f}ms/tok)  "
          f"speedup={gemv_ms/max(matmul_ms,0.001):5.1f}x")
