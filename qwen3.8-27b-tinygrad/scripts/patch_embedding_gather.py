#!/usr/bin/env python3
"""Replace the one-hot embedding reduction with a gather (exact, order-independent)."""
path = "/N/slate/demistry/tinygrad-src/tinygrad/nn/__init__.py"
with open(path) as f:
    src = f.read()

old = """def _embedding_fwd(weight:Tensor, idx:Tensor) -> Tensor:
  arange = Tensor.arange(weight.shape[0])
  return (arange == idx.unsqueeze(-1)).unsqueeze(-1).where(weight, 0).sum(-2, dtype=weight.dtype)"""

new = """def _embedding_fwd(weight:Tensor, idx:Tensor) -> Tensor:
  # gather, not a one-hot reduction: the reduction order depends on the kernel shape
  # (which depends on the token count), producing ~1e-4 fp32 differences that the
  # recurrent blocks amplify past the argmax. A gather is exact and cheaper.
  return weight[idx]"""

assert old in src, "embedding not found"
src = src.replace(old, new)
with open(path, "w") as f:
    f.write(src)
print("patched _embedding_fwd -> gather")
