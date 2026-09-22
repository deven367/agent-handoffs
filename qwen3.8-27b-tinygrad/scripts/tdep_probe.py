#!/usr/bin/env python3
"""T-dependence probe: chunked-prefill correctness (T=1 vs T=4, token 0, GDN blk0).

Setup (quartz / H100):
  salloc --account=r00117 --partition=h100-single --gres=gpu:1 --mem=128G -t 4:00:00 bash
  ssh node-quartz
  export CUDA_PATH=/N/soft/rhel8/cuda/12.6 DEV=CUDA
  python3 /tmp/tdep_probe.py
Then the full-model gate:
  python3 /tmp/compare_logits.py 1 && python3 /tmp/compare_logits.py 2 && python3 /tmp/compare_logits.py 4
Results: 2026-09-19 — items 1-5 bit-identical; scan rel 9.39e-04; item 7
crashed (4-arg shrink on 3-D tensor in finalize_after). Shapes are printed;
adjust the item-7 slice if o4full is 4-D, or run bisect_blk00_internals.py.
See docs/handoff-2026-09-19-tdep-probe.md.
"""
import sys
sys.path.insert(0, "/N/slate/demistry/tinygrad-src")
from tinygrad.llm.model import Transformer
from tinygrad.llm.kernels import amd as K
from tinygrad import Tensor
from tinygrad.uop.ops import UOp
import numpy as np

MODEL = "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
m, _ = Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
b = m.blk[0]
dev = b.attn_qkv.weight.device
print("device =", dev, flush=True)
Hk, Hv, Dk, Vd = b.num_k_heads, b.num_v_heads, b.head_k_dim, b.head_v_dim
inner = b.ssm_out.weight.shape[1]
print("dims Hk,Hv,Dk,Vd,inner =", Hk, Hv, Dk, Vd, inner, flush=True)

def diff(a, b):
  a = a.realize().numpy().reshape(-1).astype(np.float32)
  b = b.realize().numpy().reshape(-1).astype(np.float32)
  d = float(np.abs(a - b).max()); sc = max(1e-9, float(np.abs(a).max()))
  print("  max|d|=%.6e rel=%.2e" % (d, d/sc))

# deterministic non-uniform fp32 inputs, already on CUDA (slices of embedding weight)
ew = m.token_embd.weight[:, 0]
def wslice(n, off=0):
  return ew[off:off+n]

def ramp(n, shape):
  return ew[0:n].reshape(shape)
t1 = Tensor([[198]], dtype="int32", device=dev)
t4 = Tensor([[198, 248045, 846, 198]], dtype="int32", device=dev)
print("1. embedding", flush=True)
diff(m.token_embd(t1), m.token_embd(t4)[:, :1])
x = m.token_embd(t4).contiguous()
print("2. attn_qkv GEMM", flush=True)
diff(b.attn_qkv(x[:, :1].contiguous()), b.attn_qkv(x)[:, :1])
print("3. ssm_out GEMM", flush=True)
in4 = ramp(4 * inner, (1, 4, inner))
diff(b.ssm_out(in4[:, :1].contiguous()), b.ssm_out(in4)[:, :1])
print("4. qk dot (sum over Dk=128)", flush=True)
q5 = ramp(4 * Hk * Dk, (1, 4, Hk, Dk))
k5 = ramp(4 * Hk * Dk, (1, 4, Hk, Dk))
diff((q5[:, :1] * k5[:, :1]).sum(-1), (q5 * k5).sum(-1)[:, :1])
print("5. normalize(dim=-1, eps=1e-12)", flush=True)
diff(q5[:, :1].reshape(1, 1, Hk, Dk).normalize(dim=-1, eps=1e-12),
     q5.reshape(1, 4, Hk, Dk).normalize(dim=-1, eps=1e-12)[:, :1])
print("6. fused scan (token 0)", flush=True)
q4 = ramp(4 * Hv * Dk, (1, Hv, 4, Dk)).float()
k4 = ramp(4 * Hv * Dk, (1, Hv, 4, Dk)).float()
v4 = ramp(4 * Hv * Vd, (1, Hv, 4, Vd)).float()
b4 = ramp(4 * Hv, (1, Hv, 4)).float()
a4 = wslice(4 * Hv * Vd, off=4 * Hk * Dk).reshape(1, Hv, 4, Vd).float()
s0_1 = Tensor.zeros(1, Hv, Vd, Dk, device=dev)
s0_4 = Tensor.zeros(1, Hv, Vd, Dk, device=dev)
o1 = K.gated_delta_prefill(q4[:, :, :1].contiguous(), k4[:, :, :1].contiguous(),
                           v4[:, :, :1].contiguous(), b4[:, :, :1].contiguous(),
                           a4[:, :, :1].contiguous(), s0_1,
                           Tensor(UOp.variable("sp1", 0, 511).bind(0)))
o4 = K.gated_delta_prefill(q4, k4, v4, b4, a4, s0_4,
                           Tensor(UOp.variable("sp2", 0, 511).bind(0)))
diff(o1, o4[:, :, :1])
print("7. full blk0 _attention (identical input)", flush=True)
b._init_state(x)   # state attrs are lazily created; must call before zeroing
b.conv_state = b.conv_state * 0
b.recurrent_state = b.recurrent_state * 0
o4full = b._attention(x, 0)
print("o4full shape:", o4full.shape, flush=True)
# MUST realize o4full before zeroing state (it depends on conv_state/recurrent_state)
a4_np = o4full.realize().numpy().astype(np.float32)
b.conv_state = b.conv_state * 0
b.recurrent_state = b.recurrent_state * 0
o1full = b._attention(x[:, :1].contiguous(), 0)
print("o1full shape:", o1full.shape, flush=True)
a1_np = o1full.realize().numpy().astype(np.float32)

print("o4full shape:", a4_np.shape, "o1full shape:", a1_np.shape, flush=True)
sh_f, sh_o = list(a4_np.shape), list(a1_np.shape)
tdims = [i for i in range(min(len(sh_f), len(sh_o))) if sh_f[i] != sh_o[i]]
print("diff dims:", tdims, flush=True)
# Find T-axis: first dim where o4full has 4 (T=4 input)
t_axis = next((i for i in range(len(sh_f)) if sh_f[i] == 4), tdims[0] if tdims else 1)
print("T-axis at dim", t_axis, flush=True)
# Slice o4_np along T-axis to token 0
sl = [slice(None)] * len(sh_f)
sl[t_axis] = slice(None, 1, None)
a4_tok0 = a4_np[tuple(sl)].reshape(-1)
a1_flat = a1_np.reshape(-1)
d = float(np.abs(a4_tok0 - a1_flat).max())
sc = max(1e-9, float(np.abs(a1_flat).max()))
print("  max|d|=%.6e rel=%.2e" % (d, d/sc))
