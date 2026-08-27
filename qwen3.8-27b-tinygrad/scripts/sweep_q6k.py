#!/usr/bin/env python3
"""Q6_K NV kernel sweep — random integer q6 weights packed per ggml layout,
kernel vs exact-dequant f32 matmul. Run on node-lair (free NV GPU).

Pass criterion: max absolute error / max reference magnitude <= 1e-4.
Harness style mirrors /u/demistry/sweep_q4k.py (session 2).
"""
import sys
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp
from tinygrad.llm.kernels.nv_q6k import q6_k_linear, Q6_K

def pack_q6k(q, scales, d):
  """Pack q[-32,31], int8 scales, and an f16 block scale into ggml Q6_K blocks."""
  assert q.ndim == 2 and q.shape[1] == 256 and scales.shape == (q.shape[0], 16)
  codes = (q.astype(np.int16) + 32).astype(np.uint8)
  out = np.empty((q.shape[0], 210), dtype=np.uint8)
  for half in range(2):
    c = codes[:, half*128:(half+1)*128]
    out[:, half*64:(half+1)*64] = (c[:, :64] & 0xF) | ((c[:, 64:] & 0xF) << 4)
    hi = c >> 4
    out[:, 128+half*32:128+(half+1)*32] = \
      (hi[:, :32] & 3) | ((hi[:, 32:64] & 3) << 2) | ((hi[:, 64:96] & 3) << 4) | ((hi[:, 96:] & 3) << 6)
  out[:, 192:208] = scales.astype(np.int8, copy=False).view(np.uint8)
  out[:, 208:210] = np.asarray(d, dtype="<f2").reshape(1).view(np.uint8)
  return out

def dequant_q6k(block):
  """Independent scalar port of llama.cpp dequantize_row_q6_K."""
  d = np.frombuffer(block[208:210], dtype="<f2")[0].astype(np.float32)
  ql, qh = block[0:128], block[128:192]
  sc = block[192:208].astype(np.int8)
  out = np.empty(256, dtype=np.float32)
  for n in (0, 128):
    half = n // 128
    ql_base, qh_base, sc_base = half * 64, half * 32, half * 8
    for l in range(32):
      is_ = sc_base + l // 16
      q1 = int((ql[ql_base + l] & 0xF) | (((qh[qh_base + l] >> 0) & 3) << 4)) - 32
      q2 = int((ql[ql_base + l + 32] & 0xF) | (((qh[qh_base + l] >> 2) & 3) << 4)) - 32
      q3 = int((ql[ql_base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) - 32
      q4 = int((ql[ql_base + l + 32] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) - 32
      out[n + l] = d * sc[is_] * q1
      out[n + l + 32] = d * sc[is_ + 2] * q2
      out[n + l + 64] = d * sc[is_ + 4] * q3
      out[n + l + 96] = d * sc[is_ + 6] * q4
  return out


def dequant_q6k_blocks(blocks):
  """Vectorized equivalent used to build large-shape references."""
  j = np.arange(256)
  m = j % 128
  qlb = (m % 64) + (j // 128) * 64
  xl = (blocks[:, qlb] >> (4 * (m // 64))[None, :]) & 0xF
  qhb = 128 + (j // 128) * 32 + (m % 32)
  xh = (blocks[:, qhb] >> (2 * (m // 32))[None, :]) & 3
  q = (xl | (xh << 4)).astype(np.int16) - 32
  scales = blocks[:, 192:208].astype(np.int8)[:, j // 16]
  d = blocks[:, 208:210].copy().view("<f2").reshape(-1).astype(np.float32)
  return d[:, None] * q * scales

def gen_case(out_features, in_features, tokens, seed):
  assert in_features % 256 == 0
  rng = np.random.default_rng(seed)
  blocks_per_row, pattern_count = in_features // 256, min(out_features, 8)
  q = rng.integers(-32, 32, size=(pattern_count * blocks_per_row, 256), dtype=np.int16)
  scales = rng.integers(-100, 101, size=(pattern_count * blocks_per_row, 16), dtype=np.int16).astype(np.int8)
  d = np.float16(rng.uniform(0.001, 0.01))
  pattern_blocks = pack_q6k(q, scales, d)

  check_count = min(len(pattern_blocks), 8)
  recovered = np.stack([q_rec(pattern_blocks[i]) for i in range(check_count)]) - 32
  assert np.array_equal(recovered, q[:check_count]), "packer self-check failed"
  decoded = dequant_q6k_blocks(pattern_blocks)
  scalar = np.stack([dequant_q6k(pattern_blocks[i]) for i in range(check_count)])
  assert np.array_equal(decoded[:check_count], scalar), "vectorized dequant disagrees with llama.cpp port"

  weights = decoded.reshape(pattern_count, in_features)
  row_pattern = np.arange(out_features) % pattern_count
  packed = pattern_blocks.reshape(pattern_count, blocks_per_row, 210)[row_pattern].reshape(-1, 210)

  groups = in_features // 32
  k = rng.integers(-126, 127, size=(tokens, groups, 32), dtype=np.int16)
  k[:, :, 0], k[:, :, 1] = 127, -127
  unit = rng.integers(1, 5, size=(tokens, groups, 1)).astype(np.float32) * np.float32(2.0 ** -10)
  xs = (k.astype(np.float32) * unit).reshape(tokens, in_features)
  by_pattern = (xs.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float32)
  return packed, xs, by_pattern[:, row_pattern]

def q_rec(block):
  """recovered q+32 codes (6-bit) from a packed block — packer self-check helper"""
  j = np.arange(256)
  m = j % 128
  qlb = (m % 64) + (j // 128) * 64
  xl = (block[qlb] >> (4 * ((m // 64).astype(np.uint8)))) & 0xF
  qhb = 128 + (j // 128) * 32 + (m % 32)
  xh = (block[qhb] >> (2 * ((m // 32).astype(np.uint8)))) & 3
  return (xl | (xh << 4)).astype(np.int64)

def run_kernel(packed, xs, out_features, in_features, device="NV"):
  nblocks = packed.shape[0]
  raw_u8 = Tensor(packed.reshape(-1), dtype=dtypes.uint8, device=device).contiguous().realize()
  buf = raw_u8.uop.buf_uop.buffer
  raw_u16 = Tensor(UOp.from_buffer(buf.view(nblocks * 105, dtypes.uint16, 0)))
  class L: pass
  layer = L()
  layer.ggml_type = Q6_K
  layer.in_features, layer.out_features = in_features, out_features
  layer.weight, layer.bias = raw_u16, None
  xt = Tensor(xs, dtype=dtypes.float32, device=device).contiguous().realize()
  return q6_k_linear(layer, xt).realize().numpy()

def main():
  cases = [
    (256, 256, 2, 1),       # minimal, multi-token
    (512, 1024, 4, 2),      # multi-token, full warp
    (768, 768, 1, 3),       # partial warp (24 groups)
    (1024, 5120, 1, 4),     # attn_v
    (10240, 5120, 1, 5),    # attn_qkv
    (17408, 5120, 1, 6),    # ffn_down
    (248320, 5120, 1, 7),   # lm_head
  ]
  failed = 0
  for out_features, in_features, tokens, seed in cases:
    packed, xs, ref = gen_case(out_features, in_features, tokens, seed)
    y = run_kernel(packed, xs, out_features, in_features)
    y = y.reshape(tokens, out_features)
    diff = np.abs(y - ref)
    maxabs = float(diff.max())
    rel = maxabs / max(1e-8, float(np.abs(ref).max()))
    ok = rel <= 1e-4
    failed += not ok
    print(f"{'OK  ' if ok else 'FAIL'} {out_features}x{in_features} t{tokens} "
          f"maxabs={maxabs:.6g} scaled_rel={rel:.6g}")
  print(f"{'ALL OK' if not failed else f'{failed} FAILED'}")
  sys.exit(1 if failed else 0)

if __name__ == "__main__":
  main()
