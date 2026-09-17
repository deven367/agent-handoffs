#!/usr/bin/env python3
"""Fix fused RMSNorm kernel: use flat 1D output indexing."""

nv_path = "/u/demistry/tinygrad-src/tinygrad/llm/kernels/nv.py"
with open(nv_path) as f:
    src = f.read()

old_kernel = '''@functools.cache
def _rmsnorm_kernel(out:UOp, x:UOp, weight:UOp, dim:int, eps:float) -> UOp:
  B, T = x.shape[0], x.shape[1]
  bt = UOp.range(B*T, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  b, t = bt // T, bt % T
  elems = dim // 32
  # partial sum of squares
  partial = UOp.const(0, dtypes.float32)
  for i in range(elems):
    idx = lane + i*32
    v = x[b, t, idx].float()
    partial = partial + v*v
  total = _warp_reduce(partial)
  norm = (total.cast(dtypes.float32) / dim + eps).rsqrt()
  stores = []
  for i in range(elems):
    idx = lane + i*32
    v = x[b, t, idx].float()
    stores.append(out[b, t, idx].store((v * norm * weight[idx].float()).cast(out.dtype)))
  return UOp.group(*stores).end(bt, lane).sink(arg=KernelInfo(name="nv_rmsnorm", opts_to_apply=()))'''

new_kernel = '''@functools.cache
def _rmsnorm_kernel(out:UOp, x:UOp, weight:UOp, dim:int, eps:float) -> UOp:
  B, T = x.shape[0], x.shape[1]
  bt = UOp.range(B*T, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  x_flat = x.reshape(B*T, dim)
  elems = dim // 32
  # partial sum of squares
  partial = UOp.const(0, dtypes.float32)
  for i in range(elems):
    idx = lane + i*32
    v = x_flat[bt, idx].float()
    partial = partial + v*v
  total = _warp_reduce(partial)
  norm = (total.cast(dtypes.float32) / dim + eps).rsqrt()
  stores = []
  for i in range(elems):
    idx = lane + i*32
    v = x_flat[bt, idx].float()
    stores.append(out[bt*dim + idx].store((v * norm * weight[idx].float()).cast(out.dtype)))
  return UOp.group(*stores).end(bt, lane).sink(arg=KernelInfo(name="nv_rmsnorm", opts_to_apply=()))'''

assert old_kernel in src, "old kernel not found"
src = src.replace(old_kernel, new_kernel)

# Also update nv_rmsnorm to create 1D output
old_fn = '''def nv_rmsnorm(x:Tensor, weight:Tensor, eps:float=1e-6) -> Tensor:
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or isinstance(D, UOp) or D % 32 != 0:
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  out = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device)
  return Tensor.custom_kernel(out, x.contiguous(), weight.contiguous(),
                              fxn=functools.partial(_rmsnorm_kernel, dim=D, eps=eps))[0]'''

new_fn = '''def nv_rmsnorm(x:Tensor, weight:Tensor, eps:float=1e-6) -> Tensor:
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or isinstance(D, UOp) or D % 32 != 0:
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  B, T = x.shape[0], x.shape[1]
  out = Tensor.empty(B*T*D, dtype=x.dtype, device=x.device)
  result = Tensor.custom_kernel(out, x.contiguous(), weight.contiguous(),
                                fxn=functools.partial(_rmsnorm_kernel, dim=D, eps=eps))[0]
  return result.reshape(B, T, D)'''

assert old_fn in src, "old fn not found"
src = src.replace(old_fn, new_fn)

with open(nv_path, "w") as f:
    f.write(src)
print("OK")
