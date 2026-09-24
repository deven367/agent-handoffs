import functools, os, time
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.device import Device
from tinygrad.llm.kernels.nv_q4k import q4_k_linear, Q4_K
from test_coop_q4k import q4_k_linear_coop

dev = os.environ.get("DEV", "CUDA")

for outf, inf in [(1024, 5120), (5120, 5120), (27648, 5120)]:
  nblk = inf // 256
  raw_bytes = outf * nblk * 144
  raw_u32 = raw_bytes // 4
  p = Tensor.empty(raw_u32, dtype=dtypes.uint32, device=dev).realize()
  x = Tensor.empty(1, inf, dtype=dtypes.float32, device=dev).realize()
  layer = SimpleNamespace(weight=p, in_features=inf, out_features=outf, bias=None, ggml_type=Q4_K)

  @TinyJit
  def run_baseline(x):
    return q4_k_linear(layer, x).realize()

  @TinyJit
  def run_coop(x):
    return q4_k_linear_coop(layer, x).realize()

  # Warmup JIT (2 runs required to capture CUDA graph)
  for _ in range(3):
    run_baseline(x)
    run_coop(x)
  Device[dev].synchronize()

  iters = 100
  # Bench baseline
  Device[dev].synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    run_baseline(x)
  Device[dev].synchronize()
  t1 = time.perf_counter()
  base_ms = (t1 - t0) * 1000 / iters

  # Bench coop
  Device[dev].synchronize()
  t0 = time.perf_counter()
  for _ in range(iters):
    run_coop(x)
  Device[dev].synchronize()
  t1 = time.perf_counter()
  coop_ms = (t1 - t0) * 1000 / iters

  print(f"{outf:5d}x{inf:5d}: baseline={base_ms:.4f}ms ({base_ms*1000:.1f}us) | coop={coop_ms:.4f}ms ({coop_ms*1000:.1f}us) | speedup={base_ms/coop_ms:.2f}x")
