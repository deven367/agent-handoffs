#!/usr/bin/env python3
"""Fix UOp.const argument order in gated_delta_prefill."""

path = "/u/demistry/tinygrad-src/tinygrad/llm/kernels/amd.py"
with open(path) as f:
    src = f.read()

old = "start_pos=UOp.const(dtypes.int32, 1)"
new = "start_pos=UOp.const(1, dtypes.int32)"

assert old in src, "old line not found"
src = src.replace(old, new)

with open(path, "w") as f:
    f.write(src)
print("OK")
