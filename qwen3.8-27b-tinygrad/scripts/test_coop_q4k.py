import functools, os
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, UOp, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.nv import (Q8_GROUP_SIZE, _nv_dp4a, _nv_shuffle_xor, _nv_fmax, _q8_quantize)

Q4_K, GGML_BLOCK_SIZE, Q4_WORDS = 12, 256, 36

def _warp_reduce(value:UOp, maximum:bool=False) -> UOp:
  for offset in (16, 8, 4, 2, 1):
    other = _nv_shuffle_xor(value, offset)
    value = _nv_fmax(value, other) if maximum else value + other
  return value

def _load_byte(raw:UOp, base:UOp, offset:int) -> UOp:
  return (raw[base + offset//4] >> ((offset & 3)*8).cast(dtypes.uint32)) & 255

def _half(value:UOp) -> UOp:
  return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

@functools.cache
def _q4_k_coop_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  num_blocks = in_features // GGML_BLOCK_SIZE
  token_output = UOp.range(out.shape[0]*out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_output // out_features
  output = token_output % out_features

  pair = lane >> 3
  w_idx = lane & 7
  subgroup_even = pair * 2
  subgroup_odd = pair * 2 + 1

  acc = UOp.const(0, dtypes.float32)

  for b in range(num_blocks):
    base = (output * num_blocks + b) * Q4_WORDS
    w = raw[base + 4 + lane]

    grp_even = b * 8 + subgroup_even
    grp_odd = b * 8 + subgroup_odd

    x_even = xq[token, grp_even, w_idx].load()
    x_odd  = xq[token, grp_odd,  w_idx].load()

    w_even = w & 0x0f0f0f0f
    w_odd  = (w >> 4) & 0x0f0f0f0f

    dot_even = _nv_dp4a(w_even, x_even, UOp.const(0, dtypes.int32))
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x_even, UOp.const(0, dtypes.int32))

    dot_odd = _nv_dp4a(w_odd, x_odd, UOp.const(0, dtypes.int32))
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x_odd, UOp.const(0, dtypes.int32))

    sc_even = (subgroup_even < 4).where(_load_byte(raw, base, 4 + subgroup_even) & 63,
      (_load_byte(raw, base, 8 + subgroup_even) & 15) | ((_load_byte(raw, base, subgroup_even) >> 6) << 4))
    m_even = (subgroup_even < 4).where(_load_byte(raw, base, 8 + subgroup_even) & 63,
      (_load_byte(raw, base, 8 + subgroup_even) >> 4) | ((_load_byte(raw, base, 4 + subgroup_even) >> 6) << 4))

    sc_odd = (subgroup_odd < 4).where(_load_byte(raw, base, 4 + subgroup_odd) & 63,
      (_load_byte(raw, base, 8 + subgroup_odd) & 15) | ((_load_byte(raw, base, subgroup_odd) >> 6) << 4))
    m_odd = (subgroup_odd < 4).where(_load_byte(raw, base, 8 + subgroup_odd) & 63,
      (_load_byte(raw, base, 8 + subgroup_odd) >> 4) | ((_load_byte(raw, base, 4 + subgroup_odd) >> 6) << 4))

    d = _half(raw[base] & 0xffff)
    dmin = _half((raw[base] >> 16).cast(dtypes.uint32) & 0xffff)

    xd_even = xd[token, grp_even, 0]
    xd_odd  = xd[token, grp_odd,  0]

    val_even = (dot_even.float()*d*sc_even.float() - qsum_even.float()*dmin*m_even.float()) * xd_even
    val_odd  = (dot_odd.float() *d*sc_odd.float()  - qsum_odd.float() *dmin*m_odd.float() ) * xd_odd

    acc = acc + val_even + val_odd

  total = _warp_reduce(acc)
  return out[token, output, lane].store(total.cast(out.dtype)).end(token_output, lane).sink(
    arg=KernelInfo(name="nv_linear_q4_k_coop", opts_to_apply=()))

def q4_k_linear_coop(layer, x):
  assert layer.ggml_type == Q4_K and layer.in_features % GGML_BLOCK_SIZE == 0
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  out = Tensor.empty(tokens, out_features, 32, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
  kernel = _q4_k_coop_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias

# Reference packing & decoding from sweep_q4k
Q4_K_BYTES = 144
def pack_q4k(d, dmin, scales, mins, qs):
  blk = np.zeros(Q4_K_BYTES, dtype=np.uint8)
  blk[0:2] = np.frombuffer(np.float16(d).tobytes(), dtype=np.uint8)
  blk[2:4] = np.frombuffer(np.float16(dmin).tobytes(), dtype=np.uint8)
  s = np.zeros(12, dtype=np.uint8)
  s[0:4] = ((scales[0:4] & 63) | ((scales[4:8] >> 4) << 6)).astype(np.uint8)
  s[4:8] = ((mins[0:4] & 63) | ((mins[4:8] >> 4) << 6)).astype(np.uint8)
  s[8:12] = ((scales[4:8] & 0xF) | ((mins[4:8] & 0xF) << 4)).astype(np.uint8)
  blk[4:16] = s
  pairs = qs.reshape(4, 2, 32)
  blk[16:144] = (pairs[:, 0, :] | (pairs[:, 1, :] << 4)).reshape(128).astype(np.uint8)
  return blk

def decode_w(blk):
  d = np.frombuffer(blk[0:2], dtype=np.float16)[0].astype(np.float32)
  dmin = np.frombuffer(blk[2:4], dtype=np.float16)[0].astype(np.float32)
  s = blk[4:16]
  sc = np.concatenate([s[0:4] & 63, (s[8:12] & 0xF) | ((s[0:4] >> 6) << 4)]).astype(np.float32)
  mn = np.concatenate([s[4:8] & 63, (s[8:12] >> 4) | ((s[4:8] >> 6) << 4)]).astype(np.float32)
  qb = blk[16:144].reshape(4, 32)
  qs = np.stack([qb & 0xF, qb >> 4], axis=1).reshape(256).astype(np.float32)
  return (d * sc[:, None] * qs.reshape(8, 32) - dmin * mn[:, None]).reshape(256)

def test_case(outf, inf):
  nblk = inf // GGML_BLOCK_SIZE
  rng = np.random.default_rng(outf + inf * 7)
  blocks, Wc = [], np.zeros((nblk, 256), dtype=np.float32)
  for b in range(nblk):
    d = float(rng.uniform(0.01, 3.0)); dmin = float(rng.uniform(-1.0, 1.0))
    scales = rng.integers(0, 64, size=8); mins = rng.integers(0, 64, size=8)
    qs = rng.integers(0, 16, size=256)
    blk = pack_q4k(d, dmin, scales, mins, qs)
    blocks.append(blk); Wc[b] = decode_w(blk)
  raw = np.concatenate(blocks * outf)
  W = np.zeros((outf, inf), dtype=np.float32)
  for o in range(outf): W[o] = Wc.reshape(inf)
  xg = rng.integers(-32, 33, size=(inf // 32, 32), dtype=np.int16)
  xg[:, 0], xg[:, 1] = 127, -127
  x = xg.reshape(1, inf).astype(np.float32)
  p = Tensor(raw.reshape(-1).tolist(), dtype=dtypes.uint8, device=os.environ.get("DEV", "CUDA")).bitcast(dtypes.uint32).contiguous().realize()
  xt = Tensor(x.reshape(-1).tolist(), device=os.environ.get("DEV", "CUDA")).reshape(1, inf).contiguous().realize()
  layer = SimpleNamespace(weight=p, in_features=inf, out_features=outf, bias=None, ggml_type=Q4_K)
  actual = q4_k_linear_coop(layer, xt).numpy()
  expected = (x @ W.T).astype(np.float32)
  err = np.abs(actual - expected).max()
  rel = err / max(1e-30, float(np.abs(expected).max()))
  print(f"COOP out={outf:5d} in={inf:6d} maxerr={err:.6f} rel={rel:.2e} {'OK' if rel < 1e-4 else 'FAIL'}")

if __name__ == "__main__":
  for o in (1, 2, 8, 32): test_case(o, 256)
  for o in (32, 128): test_case(o, 512)
  test_case(8, 1280)
  test_case(4, 2048)
