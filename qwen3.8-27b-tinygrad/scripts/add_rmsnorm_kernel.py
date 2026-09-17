#!/usr/bin/env python3
"""Add fused RMSNorm kernel to nv.py and wire into model.py."""

# 1. Add kernel to nv.py
nv_path = "/u/demistry/tinygrad-src/tinygrad/llm/kernels/nv.py"
with open(nv_path) as f:
    nv_src = f.read()

rmsnorm_code = '''

# ******** fused RMSNorm kernel ********
@functools.cache
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
  return UOp.group(*stores).end(bt, lane).sink(arg=KernelInfo(name="nv_rmsnorm", opts_to_apply=()))

def nv_rmsnorm(x:Tensor, weight:Tensor, eps:float=1e-6) -> Tensor:
  D = x.shape[-1]
  if not nv_custom_kernels_supported(x.device) or isinstance(D, UOp) or D % 32 != 0:
    xf = x.float()
    return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * weight
  out = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device)
  return Tensor.custom_kernel(out, x.contiguous(), weight.contiguous(),
                              fxn=functools.partial(_rmsnorm_kernel, dim=D, eps=eps))[0]
'''

# Insert before the _nv_ldcs16 function (or at end)
nv_src += rmsnorm_code
with open(nv_path, "w") as f:
    f.write(nv_src)
print("nv.py: OK")

# 2. Add RMSNorm wrapper to model.py and replace nn.RMSNorm
model_path = "/u/demistry/tinygrad-src/tinygrad/llm/model.py"
with open(model_path) as f:
    model_src = f.read()

# Add import
old_import = "from tinygrad.llm.kernels.nv import nv_custom_kernels_supported"
new_import = "from tinygrad.llm.kernels.nv import nv_custom_kernels_supported, nv_rmsnorm"
assert old_import in model_src
model_src = model_src.replace(old_import, new_import)

# Add RMSNorm class after the imports section, before the config
# Find a good insertion point - after the Linear import
old_linear = "from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported"
new_linear = """from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported


class RMSNorm(nn.RMSNorm):
  def __call__(self, x:Tensor) -> Tensor:
    if nv_custom_kernels_supported(x.device):
      return nv_rmsnorm(x, self.weight, self.eps)
    return super().__call__(x)"""
assert old_linear in model_src
model_src = model_src.replace(old_linear, new_linear)

# Replace nn.RMSNorm with RMSNorm throughout
model_src = model_src.replace("nn.RMSNorm", "RMSNorm")

with open(model_path, "w") as f:
    f.write(model_src)
print("model.py: OK")
