#!/usr/bin/env python3
"""Check: does tinygrad's fp16 matmul use tensor cores (WMMA) on CUDA?"""
import sys, time
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops
from tinygrad.helpers import GlobalCounters

# Test fp16 matmul
a = Tensor.randn(256, 5120, dtype=dtypes.float16).realize()
b = Tensor.randn(5120, 5120, dtype=dtypes.float16).realize()
c = (a @ b).realize()

# Check the linear IR for WMMA ops
linear = Tensor.linear_with_vars(c)[0]
ops = {}
for u in linear.toposort():
    ops[u.op.name] = ops.get(u.op.name, 0) + 1
print("ops in matmul linear:", {k:v for k,v in sorted(ops.items()) if v > 2})

# Check the scheduled kernels
from tinygrad.engine.realize import compile_linear
kernels = []
for u in linear.toposort():
    if u.op is Ops.CALL:
        kernels.append(u)
print(f"CALL ops (kernels): {len(kernels)}")

# Benchmark
for _ in range(3): (a @ b).realize()
GlobalCounters.reset()
t0 = time.perf_counter()
for _ in range(10): c = (a @ b).realize()
t1 = time.perf_counter()
ms = (t1-t0)/10*1000
flops = 2 * 256 * 5120 * 5120
print(f"\n256x5120 @ 5120x5120 fp16: {ms:.3f} ms -> {flops/ms/1e9:.1f} TFLOPS")
print(f"(H100 fp16 TC peak ~990 TFLOPS, fp32 ~67 TFLOPS)")
