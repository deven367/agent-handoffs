#!/usr/bin/env python3
"""Patch gated_delta_prefill: resolve start_pos at capture time, avoid kernel_var binding issue."""

path = "/u/demistry/tinygrad-src/tinygrad/llm/kernels/amd.py"
with open(path) as f:
    src = f.read()

old_body = """  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  if start_pos is None:
    srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
    return Tensor.custom_kernel(*srcs, fxn=functools.partial(_gated_delta_prefill_kernel, nv=nv_device))[0]
  # pass start_pos as an explicit source tensor so the JIT binds it correctly
  sp_var = start_pos.uop.unbind_all()[0]
  sp_tensor = Tensor(sp_var).reshape(1)
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq, sp_tensor)
  result = Tensor.custom_kernel(*srcs, fxn=functools.partial(_gated_delta_prefill_kernel, start_pos=sp_var, nv=nv_device))
  return result[0]"""

new_body = """  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  if start_pos is None: return Tensor.custom_kernel(*srcs, fxn=functools.partial(_gated_delta_prefill_kernel, nv=nv_device))[0]
  # resolve start_pos at capture time: the JIT captures separate graphs for prefill
  # (start_pos=0, reset state) and rollout (start_pos>0, keep state). kernel_var
  # binding doesn't propagate on CUDA, so we branch here instead.
  from tinygrad.uop.ops import resolve
  sp_val = start_pos.uop.bindings[start_pos.uop.unbind_all()[0]]
  if resolve(sp_val == 0):
    # prefill: always reset state (start_pos=0 → initial=True)
    return Tensor.custom_kernel(*srcs, fxn=functools.partial(_gated_delta_prefill_kernel, start_pos=None, nv=nv_device))[0]
  else:
    # rollout: never reset (start_pos>0 → initial=False)
    return Tensor.custom_kernel(*srcs, fxn=functools.partial(_gated_delta_prefill_kernel, start_pos=UOp.const(dtypes.int32, 1), nv=nv_device))[0]"""

assert old_body in src, f"old body not found"
src = src.replace(old_body, new_body)

with open(path, "w") as f:
    f.write(src)
print("OK")
