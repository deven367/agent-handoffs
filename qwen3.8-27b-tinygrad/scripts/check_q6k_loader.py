#!/usr/bin/env python3
"""Validate tinygrad's current Q6_K loader against real model bytes.

Compares:
  (a) tinygrad.llm.gguf.ggml_data_to_tensor, executed directly
  (b) a source-equivalent numpy formula
  (c) an independent numpy port of llama.cpp dequantize_row_q6_K
  (d) the hypothetical unshifted-high-bits formula, which must disagree

usage: check_q6k_loader.py [path-to-gguf]
"""
import struct, sys, time
import numpy as np

GGUF = sys.argv[1] if len(sys.argv) > 1 else "/scratch/local/demistry/models/Qwen3.8-27B-OBLITERATED.Q4_K_M.gguf"

def open_retry(path, tries=60, delay=5.0):
  for i in range(tries):
    try:
      f = open(path, "rb")
      f.read(4)
      f.seek(0)
      print(f"[open ok on attempt {i+1}]")
      return f
    except FileNotFoundError:
      print(f"[open: ENOENT, retry {i+1}/{tries} in {delay}s]")
      time.sleep(delay)
  raise RuntimeError("file never appeared")

def gguf_tensors(f):
  scalar_fmts = {0:"B", 1:"b", 2:"H", 3:"h", 4:"I", 5:"i", 6:"f", 7:"?", 10:"Q", 11:"q", 12:"d"}

  def unpack(fmt):
    size = struct.calcsize("<" + fmt)
    raw = f.read(size)
    if len(raw) != size: raise EOFError(f"short GGUF read: wanted {size} bytes, got {len(raw)}")
    return struct.unpack("<" + fmt, raw)[0]

  def read_str(keep=True):
    n = unpack("Q")
    raw = f.read(n)
    if len(raw) != n: raise EOFError(f"short GGUF string: wanted {n} bytes, got {len(raw)}")
    return raw.decode() if keep else None

  def read_val(typ, keep=False):
    if typ in scalar_fmts:
      fmt = scalar_fmts[typ]
      value = unpack(fmt)
      return value if keep else None
    if typ == 8: return read_str(keep)
    if typ == 9:
      elem_typ, count = unpack("I"), unpack("Q")
      if not keep and elem_typ in scalar_fmts:
        f.seek(struct.calcsize("<" + scalar_fmts[elem_typ]) * count, 1)
        return None
      values = [read_val(elem_typ, keep) for _ in range(count)]
      return values if keep else None
    raise ValueError(f"unsupported GGUF metadata type {typ}")

  f.seek(0)
  magic = f.read(4)
  version, n_tensors, n_kv = unpack("I"), unpack("Q"), unpack("Q")
  assert magic == b"GGUF" and version in (2, 3), (magic, version)
  alignment = 32
  for _ in range(n_kv):
    key = read_str()
    value = read_val(unpack("I"), key == "general.alignment")
    if key == "general.alignment": alignment = int(value)
  tensors = []
  for _ in range(n_tensors):
    name = read_str()
    dims = tuple(unpack("Q") for _ in range(unpack("I")))
    typ, off = unpack("I"), unpack("Q")
    tensors.append((name, dims, typ, off))
  data_start = (f.tell() + alignment - 1) // alignment * alignment
  return [(name, dims, typ, data_start + off) for name, dims, typ, off in tensors]

def read_blocks(f, off, n_blocks=64):
  f.seek(off)
  raw = f.read(n_blocks * 210)
  if len(raw) != n_blocks * 210: raise EOFError(f"short Q6_K read: wanted {n_blocks * 210} bytes, got {len(raw)}")
  return np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 210)

def dequant_q6k_ggml(block: np.ndarray) -> np.ndarray:
  """Direct port of llama.cpp ggml-quants.c dequantize_row_q6_K."""
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
      out[n + l]      = d * sc[is_] * q1
      out[n + l + 32] = d * sc[is_ + 2] * q2
      out[n + l + 64] = d * sc[is_ + 4] * q3
      out[n + l + 96] = d * sc[is_ + 6] * q4
  return out

def dequant_q6k_formula(block: np.ndarray, shift_high: bool) -> np.ndarray:
  """Numpy port of tinygrad's tensor layout; shift_high=False models the suspected bug."""
  d = np.frombuffer(block[208:210], dtype="<f2")[0].astype(np.float32)
  sc = block[192:208].astype(np.int8)
  j = np.arange(256)
  qb = (j % 128) % 64 + (j // 128) * 64
  nib = (j % 128) // 64
  xl = (block[qb] >> (4 * nib)) & 0xF
  hb = 128 + (j // 128) * 32 + (j % 32)
  slot = (j % 128) // 32
  xh = (block[hb] >> (2 * slot)) & 3
  combined = xl | (xh << 4 if shift_high else xh)
  q = combined.astype(np.int16) - 32
  return (d * q * sc[(j // 16).astype(np.int64)]).astype(np.float32)


def dequant_q6k_tinygrad(blocks: np.ndarray) -> np.ndarray:
  from tinygrad import Tensor, dtypes
  from tinygrad.llm.gguf import ggml_data_to_tensor
  raw = Tensor(blocks.reshape(-1).copy(), dtype=dtypes.uint8)
  return ggml_data_to_tensor(raw, blocks.shape[0] * 256, 14).numpy().reshape(-1).astype(np.float32)

def main():
  with open_retry(GGUF) as f:
    tensors = gguf_tensors(f)
    q6k = [(n, d, o) for n, d, t, o in tensors if t == 14]
    print(f"Q6_K tensors: {len(q6k)}")
    if not q6k: raise RuntimeError("model has no Q6_K tensors")
    picks = [t for t in q6k if "output.weight" in t[0]]
    for tensor in q6k:
      if len(picks) >= 3: break
      if tensor not in picks: picks.append(tensor)
    for name, dims, off in picks:
      n_el = int(np.prod(dims))
      blocks = read_blocks(f, off, min(64, n_el // 256))
      ref = np.stack([dequant_q6k_ggml(b) for b in blocks]).ravel()
      actual = dequant_q6k_tinygrad(blocks)
      source_formula = np.stack([dequant_q6k_formula(b, True) for b in blocks]).ravel()
      unshifted = np.stack([dequant_q6k_formula(b, False) for b in blocks]).ravel()
      def report(tag, x):
        diff = np.abs(x - ref)
        rel = (diff / (np.abs(ref) + 1e-8)).max()
        print(f"  {tag}: maxabs={diff.max():.6g} maxrel={rel:.6g} exact={bool(np.array_equal(x, ref))}")
      print(f"{name} {dims} ({n_el} el, {n_el//256} blocks, first {len(blocks)} blocks)")
      report("tinygrad actual             ", actual)
      report("source formula (xh << 4)    ", source_formula)
      report("hypothetical unshifted xh   ", unshifted)
      print(f"  ref stats: min={ref.min():.4g} max={ref.max():.4g} mean|x|={np.abs(ref).mean():.4g}")
      if not np.array_equal(actual, ref): raise AssertionError(f"tinygrad loader disagrees with ggml for {name}")
      if not np.array_equal(source_formula, ref): raise AssertionError(f"source formula disagrees with ggml for {name}")
      if np.array_equal(unshifted, ref): raise AssertionError(f"unshifted formula unexpectedly agrees for {name}")
  print("Q6_K LOADER CHECK: current tinygrad == ggml port (exact)")


if __name__ == "__main__":
  main()
