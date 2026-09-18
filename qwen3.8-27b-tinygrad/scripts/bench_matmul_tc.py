#!/usr/bin/env python3
"""Benchmark fp16 matmul with different TC settings."""
import sys, time, os
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters

M, K, N = 512, 5120, 5120
a = Tensor.randn(M, K, dtype=dtypes.float16).realize()
b = Tensor.randn(K, N, dtype=dtypes.float16).realize()

for _ in range(3): c = (a @ b).realize()
GlobalCounters.reset()
t0 = time.perf_counter()
for _ in range(10): c = (a @ b).realize()
t1 = time.perf_counter()
ms = (t1-t0)/10*1000
flops = 2 * M * K * N
print(f"M={M} K={K} N={N}: {ms:.3f} ms -> {flops/ms/1e9:.1f} TFLOPS  env(TC={os.getenv('TC','1')} TC_OPT={os.getenv('TC_OPT','0')} BEAM={os.getenv('BEAM','0')})")
