import functools, os, time
from types import SimpleNamespace
import numpy as np
from tinygrad import Tensor, UOp, dtypes, Device, TinyJit
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.llm.kernels.nv import (Q8_GROUP_SIZE, _nv_dp4a, _nv_shuffle_xor, _nv_fmax, _q8_quantize)
from test_coop_q4k import pack_q4k, decode_w, q4_k_linear_coop

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

# Variant 1: 16 threads per block, 2 words per thread (uint2, 64-bit loads)
@functools.cache
def _q4_k_v2_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  num_blocks = in_features // GGML_BLOCK_SIZE
  assert num_blocks % 2 == 0, f"num_blocks must be even, got {num_blocks}"
  token_output = UOp.range(out.shape[0]*out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_output // out_features
  output = token_output % out_features

  lane16 = lane & 15
  block_in_pair = lane >> 4
  pair = lane16 >> 2
  k = lane16 & 3
  subgroup_even = pair * 2
  subgroup_odd = pair * 2 + 1

  w0_idx = k * 2
  w1_idx = k * 2 + 1

  acc = UOp.const(0, dtypes.float32)

  for b_pair in range(num_blocks // 2):
    b = b_pair * 2 + block_in_pair
    base = (output * num_blocks + b) * Q4_WORDS

    raw_idx = base + 4 + pair * 8 + k * 2
    w0 = raw[raw_idx]
    w1 = raw[raw_idx + 1]

    grp_even = b * 8 + subgroup_even
    grp_odd = b * 8 + subgroup_odd

    x0_even = xq[token, grp_even, w0_idx].load()
    x1_even = xq[token, grp_even, w1_idx].load()
    x0_odd  = xq[token, grp_odd,  w0_idx].load()
    x1_odd  = xq[token, grp_odd,  w1_idx].load()

    w0_even = w0 & 0x0f0f0f0f
    w0_odd  = (w0 >> 4) & 0x0f0f0f0f
    w1_even = w1 & 0x0f0f0f0f
    w1_odd  = (w1 >> 4) & 0x0f0f0f0f

    dot_even = _nv_dp4a(w0_even, x0_even, UOp.const(0, dtypes.int32))
    dot_even = _nv_dp4a(w1_even, x1_even, dot_even)
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0_even, UOp.const(0, dtypes.int32))
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1_even, qsum_even)

    dot_odd = _nv_dp4a(w0_odd, x0_odd, UOp.const(0, dtypes.int32))
    dot_odd = _nv_dp4a(w1_odd, x1_odd, dot_odd)
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0_odd, UOp.const(0, dtypes.int32))
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1_odd, qsum_odd)

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
    arg=KernelInfo(name="nv_linear_q4_k_v2", opts_to_apply=()))

# Variant 2: 8 threads per block, 4 words per thread (uint4, 128-bit loads)
@functools.cache
def _q4_k_v4_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int) -> UOp:
  num_blocks = in_features // GGML_BLOCK_SIZE
  assert num_blocks % 4 == 0, f"num_blocks must be divisible by 4, got {num_blocks}"
  token_output = UOp.range(out.shape[0]*out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  token = token_output // out_features
  output = token_output % out_features

  lane8 = lane & 7
  block_in_quad = lane >> 3
  pair = lane8 >> 1
  k = lane8 & 1
  subgroup_even = pair * 2
  subgroup_odd = pair * 2 + 1

  w0_idx = k * 4
  w1_idx = k * 4 + 1
  w2_idx = k * 4 + 2
  w3_idx = k * 4 + 3

  acc = UOp.const(0, dtypes.float32)

  for b_quad in range(num_blocks // 4):
    b = b_quad * 4 + block_in_quad
    base = (output * num_blocks + b) * Q4_WORDS

    raw_idx = base + 4 + pair * 8 + k * 4
    w0 = raw[raw_idx]
    w1 = raw[raw_idx + 1]
    w2 = raw[raw_idx + 2]
    w3 = raw[raw_idx + 3]

    grp_even = b * 8 + subgroup_even
    grp_odd = b * 8 + subgroup_odd

    x0_even = xq[token, grp_even, w0_idx].load()
    x1_even = xq[token, grp_even, w1_idx].load()
    x2_even = xq[token, grp_even, w2_idx].load()
    x3_even = xq[token, grp_even, w3_idx].load()

    x0_odd  = xq[token, grp_odd,  w0_idx].load()
    x1_odd  = xq[token, grp_odd,  w1_idx].load()
    x2_odd  = xq[token, grp_odd,  w2_idx].load()
    x3_odd  = xq[token, grp_odd,  w3_idx].load()

    w0_even = w0 & 0x0f0f0f0f
    w0_odd  = (w0 >> 4) & 0x0f0f0f0f
    w1_even = w1 & 0x0f0f0f0f
    w1_odd  = (w1 >> 4) & 0x0f0f0f0f
    w2_even = w2 & 0x0f0f0f0f
    w2_odd  = (w2 >> 4) & 0x0f0f0f0f
    w3_even = w3 & 0x0f0f0f0f
    w3_odd  = (w3 >> 4) & 0x0f0f0f0f

    dot_even = _nv_dp4a(w0_even, x0_even, UOp.const(0, dtypes.int32))
    dot_even = _nv_dp4a(w1_even, x1_even, dot_even)
    dot_even = _nv_dp4a(w2_even, x2_even, dot_even)
    dot_even = _nv_dp4a(w3_even, x3_even, dot_even)

    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0_even, UOp.const(0, dtypes.int32))
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1_even, qsum_even)
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x2_even, qsum_even)
    qsum_even = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x3_even, qsum_even)

    dot_odd = _nv_dp4a(w0_odd, x0_odd, UOp.const(0, dtypes.int32))
    dot_odd = _nv_dp4a(w1_odd, x1_odd, dot_odd)
    dot_odd = _nv_dp4a(w2_odd, x2_odd, dot_odd)
    dot_odd = _nv_dp4a(w3_odd, x3_odd, dot_odd)

    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x0_odd, UOp.const(0, dtypes.int32))
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x1_odd, qsum_odd)
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x2_odd, qsum_odd)
    qsum_odd = _nv_dp4a(UOp.const(0x01010101, dtypes.uint32), x3_odd, qsum_odd)

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
    arg=KernelInfo(name="nv_linear_q4_k_v4", opts_to_apply=()))

def q4_k_linear_opt(layer, x):
  tokens = int(x.numel()) // layer.in_features
  raw, out_features, in_features = layer.weight.uop.buf_uop, layer.out_features, layer.in_features
  xq, xd = _q8_quantize(x, tokens, in_features)
  out = Tensor.empty(tokens, out_features, 32, dtype=dtypes.float32, device=x.device).uop
  all_srcs = (out, raw, xq.uop, xd.uop)
  params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(all_srcs))
  num_blocks = in_features // GGML_BLOCK_SIZE
  if num_blocks % 4 == 0:
    kernel = _q4_k_v4_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  elif num_blocks % 2 == 0:
    kernel = _q4_k_v2_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  else:
    from tinygrad.llm.kernels.nv_q4k import _q4_k_decode_kernel
    kernel = _q4_k_decode_kernel(*params, out_features=out_features, in_features=in_features).call(*all_srcs)
  result = Tensor(out.after(kernel))[..., 0]
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias

def test_correctness():
  dev = os.environ.get("DEV", "CUDA")
  for outf in (1, 2, 8, 32):
    for inf in (256 * 2, 512, 1024, 1280, 2048):
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
      p = Tensor(raw.reshape(-1).tolist(), dtype=dtypes.uint8, device=dev).bitcast(dtypes.uint32).contiguous().realize()
      xt = Tensor(x.reshape(-1).tolist(), device=dev).reshape(1, inf).contiguous().realize()
      layer = SimpleNamespace(weight=p, in_features=inf, out_features=outf, bias=None, ggml_type=Q4_K)

      act_coop = q4_k_linear_coop(layer, xt).numpy()
      act_opt  = q4_k_linear_opt(layer, xt).numpy()
      expected = (x @ W.T).astype(np.float32)

      diff = np.abs(act_opt - act_coop).max()
      err  = np.abs(act_opt - expected).max()
      rel  = err / max(1e-30, float(np.abs(expected).max()))
      status = "OK" if rel < 1e-4 else "FAIL"
      ver = "v4 (uint4)" if nblk % 4 == 0 else "v2 (uint2)"
      print(f"OPT [{ver}] out={outf:4d} in={inf:5d} | opt_vs_coop_diff={diff:.6e} maxerr={err:.6f} rel={rel:.2e} {status}")
      assert status == "OK", f"Mismatch: rel={rel}, diff={diff}"

if __name__ == "__main__":
  test_correctness()
