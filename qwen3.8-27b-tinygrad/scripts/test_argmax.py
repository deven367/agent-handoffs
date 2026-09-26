#!/usr/bin/env python3
"""Unit sweep and tie-breaking tests for custom two-stage warp argmax (nv_argmax)."""
import os, sys, numpy as np

# Portable search for tinygrad-src
for p in [os.path.expanduser("~/projects/tinygrad-src"), "/u/demistry/tinygrad-src", "/N/slate/demistry/tinygrad-src"]:
  if os.path.isdir(p) and p not in sys.path:
    sys.path.append(p)

from tinygrad import Tensor, dtypes
from tinygrad.llm.kernels.nv import nv_argmax, nv_custom_kernels_supported

def test_nv_argmax():
  device = "CUDA"
  if not nv_custom_kernels_supported(device):
    print("CUDA/NV custom kernels not supported on this platform, skipping.")
    return

  print("Running nv_argmax unit sweep across vocabulary sizes...")
  np.random.seed(42)

  # Test across representative vocabulary sizes
  # 248320 (Qwen3.8-27B), 151936 (Qwen2.5-0.5B), 128256 (Llama 3), 32000 (Llama 2)
  for N in [32000, 128256, 151936, 248320]:
    tokens = 1
    for trial in range(5):
      x_np = (np.random.randn(tokens, N) * 1e4).astype(np.float32)
      expected = int(x_np[0].argmax())

      x = Tensor(x_np, device=device)
      res = nv_argmax(x, keepdim=True)
      got = int(res.numpy()[0, 0])
      assert got == expected, f"N={N} trial {trial} mismatch: got {got}, expected {expected}"
    print(f"OK   vocab={N:6d} (5 trials match numpy argmax bit-exact)")

  # Test constructed tie-breaking (earliest/lowest flat index must win)
  print("Testing tie-breaking rules...")
  N = 248320
  x_np = np.zeros((1, N), dtype=np.float32)
  x_np[0, 500] = 100.0
  x_np[0, 2000] = 100.0
  x = Tensor(x_np, device=device)
  res = nv_argmax(x, keepdim=True)
  got = int(res.numpy()[0, 0])
  assert got == 500, f"Tie breaking failed: got {got}, expected 500"
  print("OK   tie-break (500 vs 2000 => 500)")

  # All-equal tie breaking (lane 0 / index 0 must win)
  x_np = np.full((1, N), 42.0, dtype=np.float32)
  x = Tensor(x_np, device=device)
  res = nv_argmax(x, keepdim=True)
  got = int(res.numpy()[0, 0])
  assert got == 0, f"All-equal tie breaking failed: got {got}, expected 0"
  print("OK   all-equal (index 0 wins)")

  print("ALL OK")

if __name__ == "__main__":
  test_nv_argmax()
