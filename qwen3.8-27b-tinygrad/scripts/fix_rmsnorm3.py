#!/usr/bin/env python3
"""Fix RMSNorm: use 2D output (batch, dim) to avoid coalesce pass issues."""

nv_path = "/u/demistry/tinygrad-src/tinygrad/llm/kernels/nv.py"
with open(nv_path) as f:
    src = f.read()

old_kernel = '''@functools.cache
def _rmsnorm_kernel(out:UOp, x:UOp, weight:UOp, dim:int, eps:float, batch:int) -> UOp:
  gid = UOp.range(batch, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  elems = dim // 32
  # partial sum of squares
  partial = UOp.const(0, dtypes.float32)
  for i in range(elems):
    idx = lane + i*32
    v = x[gid*dim + idx].float()
    partial = partial + v*v
  total = _warp_reduce(partial)
  norm = (total.cast(dtypes.float32) / dim + eps).rsqrt()
  stores = []
  for i in range(elems):
    idx = lane + i*32
    v = x[gid*dim + idx].float()
    stores.append(out[gid*dim + idx].store((v * norm * weight[idx].float()).cast(out.dtype)))
  return UOp.group(*stores).end(gid, lane).sink(arg=KernelInfo(name="nv_rmsnorm", opts_to_apply=()))'''

new_kernel = '''@functools.cache
def _rmsnorm_kernel(out:UOp, x:UOp, weight:UOp, dim:int, eps:float, batch:int) -> UOp:
  gid = UOp.range(batch, 0, AxisType.GLOBAL)
  lane = UOp.range(32, 1, AxisType.LOCAL)
  elems = dim // 32
  x2d = x.reshape(batch, dim)
  # partial sum of squares
  partial = UOp.const(0, dtypes.float32)
  for i in range(elems):
    col = lane + i*32
    v = x2d[gid, col].float()
    partial = partial + v*v
  total = _warp_reduce(partial)
  norm = (total.cast(dtypes.float32) / dim + eps).rsqrt()
  stores = []
  for i in range(elems):
    col = lane + i*32
    v = x2d[gid, col].float()
    stores.append(out[gid, col].store((v * norm * weight[col].float()).cast(out.dtype)))
  return UOp.group(*stores).end(gid, lane).sink(arg=KernelInfo(name="nv_rmsnorm", opts_to_apply=()))'''

assert old_kernel in src, "old kernel not found"
src = src.replace(old_kernel, new_kernel)

old_fn = '''def nv_rmsnorm(x:Tensor, weight:Tensor, eps:float=1e-6) -> Tensor:
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or isinstance(D, UOp) or D % 32 != 0:
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  orig_shape = x.shape
  x_flat = x.reshape(-1, D)
  batch = x_flat.shape[0]
  out = Tensor.empty(batch*D, dtype=x.dtype, device=x.device)
  result = Tensor.custom_kernel(out, x_flat.contiguous(), weight.contiguous(),
                                fxn=functools.partial(_rmsnorm_kernel, dim=D, eps=eps, batch=batch))[0]
  return result.reshape(orig_shape)'''

new_fn = '''def nv_rmsnorm(x:Tensor, weight:Tensor, eps:float=1e-6) -> Tensor:
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or isinstance(D, UOp) or D % 32 != 0:
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  orig_shape = x.shape
  x_flat = x.reshape(-1, D)
  batch = x_flat.shape[0]
  out = Tensor.empty(batch, D, dtype=x.dtype, device=x.device)
  result = Tensor.custom_kernel(out, x_flat.contiguous(), weight.contiguous(),
                                fxn=functools.partial(_rmsnorm_kernel, dim=D, eps=eps, batch=batch))[0]
  return result.reshape(orig_shape)'''

assert old_fn in src, "old fn not found"
src = src.replace(old_fn, new_fn)

with open(nv_path, "w") as f:
    f.write(src)
print("OK")
