#!/usr/bin/env python3
"""Patch gated_delta_prefill: resolve start_pos at capture time."""

path = "/u/demistry/tinygrad-src/tinygrad/llm/kernels/amd.py"
with open(path) as f:
    src = f.read()

old = "  sp_val = start_pos.uop.bindings[start_pos.uop.unbind_all()[0]]"
new = "  _, sp_val = start_pos.uop.unbind()"

assert old in src, "old line not found"
src = src.replace(old, new)

with open(path, "w") as f:
    f.write(src)
print("OK")
