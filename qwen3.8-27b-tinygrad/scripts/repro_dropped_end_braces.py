#!/usr/bin/env python3
"""Repro: cstyle emits a `for (...) {` with no matching `}` when a RANGE's END uop
is not reachable from the SINK.

Context: found on the qwen27b-nv-q8-kernel fork as
    NVRTC_ERROR_COMPILATION / At end of source: error: expected a "}"
with 8 opening braces and 7 closing braces in the generated CUDA (see
docs/progress.md, "2026-09-06 - Prefill investigation"). The fork carries a
5-line safety net in CStyleLanguage._render (39a790966); upstream does not.

Run from a tinygrad checkout root (no GPU, no CUDA needed):
    DEV=CPU python3 repro_dropped_end_braces.py
"""
import sys
import pathlib
from tinygrad import Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.dtype import dtypes
from tinygrad.codegen.late.linearizer import linearize

REN = Device["CPU"].renderer  # any CStyleLanguage renders this way


def build_linear() -> list[UOp]:
  """A loop whose body STORE is reachable through AFTER, but whose END is not.

  `after = buf.after(store)` references the STORE directly; nothing references the
  END, so linearize() (which walks src edges from the SINK) drops the END while the
  STORE - and therefore the RANGE it is indexed by - stays in the list.
  """
  buf = UOp.placeholder((8,), dtypes.int, slot=0)
  nbuf = UOp.placeholder((1,), dtypes.int, slot=1)
  bound = nbuf.index(UOp.const(0, dtypes.int)).load()          # runtime loop bound
  rng = UOp.range(bound, 0, AxisType.WEAK, dtype=dtypes.int)
  store = buf.index(rng).store(UOp.const(1, dtypes.int))
  after = buf.after(store)
  val = after.index(UOp.const(0, dtypes.int)).load()
  return linearize(UOp.sink(val, arg=KernelInfo(name="repro_dropped_end")))


def brace_balance(src: str) -> int:
  return src.count("{") - src.count("}")


def clamped(cls):
  """The fork's 5-line safety net, applied to the live renderer instance."""
  class Clamped(cls):
    def _render(self, uops):
      name, kernel, bufs = super()._render(uops)
      depth = 1 + sum(u.op in (Ops.IF, Ops.RANGE) for u in uops) - sum(u.op in (Ops.END, Ops.ENDIF) for u in uops)
      while depth > 1:
        depth -= 1
        kernel.append("  "*depth + "}")
      return name, kernel, bufs
  inst = Clamped.__new__(Clamped)
  inst.__dict__.update(REN.__dict__)  # instance attrs only (compiler/target); never copy a class __dict__
  return inst


lin = build_linear()
n_ranges = sum(u.op is Ops.RANGE for u in lin)
n_ends = sum(u.op is Ops.END for u in lin)

print(f"RANGE uops in linear list: {n_ranges}")
print(f"END   uops in linear list: {n_ends}")
assert n_ranges == 1 and n_ends == 0, "expected the RANGE to survive and the END to be dropped"

src = REN.render(lin)
bal = brace_balance(src)
print(f"brace balance without the safety net: {bal}  (<- unclosed '{{' means malformed source)")
print("-" * 60)
print(src)
print("-" * 60)

p = pathlib.Path("/tmp/repro_dropped_end.c")
p.write_text(src)
print(f"wrote {p} for an independent compiler check, e.g.: clang -x c -fsyntax-only {p}")

fixed = clamped(type(REN)).render(lin)
print(f"brace balance with the fork's safety net: {brace_balance(fixed)}")

assert bal > 0, "repro did not trigger: END was not dropped"
assert brace_balance(fixed) == 0, "safety net did not restore brace balance"
print("REPRODUCED: malformed source from a dropped END uop")
