#!/usr/bin/env python3
"""Patch gated_delta_prefill to use custom_kernel with start_pos as explicit source."""
import sys

path = "/u/demistry/tinygrad-src/tinygrad/llm/kernels/amd.py"
with open(path) as f:
    src = f.read()

old_body = """  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  if start_pos is None: return Tensor.custom_kernel(*srcs, fxn=_gated_delta_prefill_kernel, nv=nv_device)[0]
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  assert start_pos.uop.is_bound_var
  # the bound start_pos reaches the graph through the state AFTER chain, like the flash kernels' valid_end
  call = _gated_delta_prefill_kernel(*params, kernel_var(start_pos.uop.src[0]), nv=nv_device).call(*contig)
  return Tensor(contig[0].after(call))"""

new_body = """  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  if start_pos is None:
    srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
    return Tensor.custom_kernel(*srcs, fxn=functools.partial(_gated_delta_prefill_kernel, nv=nv_device))[0]
  # pass start_pos as an explicit source tensor so the JIT binds it correctly
  sp_var = start_pos.uop.unbind_all()[0]
  sp_tensor = Tensor(sp_var).reshape(1)
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq, sp_tensor)
  result = Tensor.custom_kernel(*srcs, fxn=functools.partial(_gated_delta_prefill_kernel, start_pos=sp_var, nv=nv_device))
  return result[0]"""

assert old_body in src, "old body not found"
src = src.replace(old_body, new_body)

with open(path, "w") as f:
    f.write(src)
print("OK")
