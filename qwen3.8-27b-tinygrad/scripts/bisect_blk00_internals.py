#!/usr/bin/env python3
"""Instrument blk00 GatedDeltaNetBlock._attention: find first diverging intermediate.

Preserves @function(precompile=True) for block 0 — intermediates are returned as
additional @function outputs (CALL UOp gettuple), making them realizable without
changing compilation behaviour. Stashes key intermediates at each stage of the
data flow, then compares stats between two chunk sizes. The first diverging
intermediate names the buggy subsystem.

Usage (on node-lair):
  DEV=CUDA python3 bisect_blk00_internals.py "[1]" 1 2 [tol]
"""
import sys, functools
import numpy as np
import tinygrad.llm.model as m
from tinygrad.llm.model import Transformer, GatedDeltaNetBlock
from tinygrad import Tensor, function
from tinygrad.uop.ops import UOp

MODEL = "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"

# Populated during _attention tracing: list of (name, time_dim).
# time_dim=-1 means no time dimension (don't shrink).
META: list[tuple[str, int]] = []
_call_count = [0]
_orig_call = GatedDeltaNetBlock.__call__


def _instrumented_attention(self, x: Tensor, start_pos, pre_norm_x: Tensor | None = None) -> tuple[Tensor, list[Tensor]]:
    """Exact copy of GatedDeltaNetBlock._attention. Returns (result, intermediates).

    Intermediates are appended in data-flow order; META records their names/dims
    in the same order so the caller can match them after @function returns.
    """
    B, T, _ = x.shape
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]
    inter: list[Tensor] = []

    def rec(name, t, td):
        META.append((name, td)); inter.append(t)

    if pre_norm_x is not None: rec("pre_norm_x", pre_norm_x, 1)

    rec("x_in", x, 1)

    # input processing
    x = x.half()
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, T, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    log_alpha = ((alpha.float() + self.ssm_dt["bias"]).softplus().reshape(B, T, self.num_v_heads, -1) *
                 self.ssm_a.reshape(self.num_v_heads, -1))

    # qkv conv
    conv_state = initial.where(0, self.conv_state)
    win = Tensor.zeros(B, self.ssm_conv_kernel-1 + T_pad, self.conv_channels).uop
    win = win.after(win[:, :self.ssm_conv_kernel-1].store(conv_state.cast(win.dtype).uop))
    win = win.after(win[:, self.ssm_conv_kernel-1:self.ssm_conv_kernel-1+T].store(self.attn_qkv(x).cast(win.dtype).uop))
    conv_window = Tensor(win)
    conv_state_store = self.conv_state.uop.store(conv_window[:, T:T+self.ssm_conv_kernel-1].cast(self.conv_state.dtype).uop)
    conv_out = functools.reduce(lambda a,b: a+b,
      (conv_window[:, i:i+T_pad] * self.ssm_conv1d["weight"][:, i] for i in range(self.ssm_conv_kernel))).silu()
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))

    rec("conv_out", conv_out, 1)
    rec("beta_pad", beta, 1)
    rec("log_alpha_pad", log_alpha, 1)

    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    qk_eps = 1e-12 if is_kda else 1e-6
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim).normalize(dim=-1, eps=qk_eps)
            .repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()

    rec("q", q, 2)
    rec("k", k, 2)
    rec("v", v, 2)
    rec("alpha", alpha, 2)
    rec("beta_tx", beta, 2)

    # recurrent scan (non-AMD path — CUDA never takes the fused kernel branch)
    state = Tensor(self.recurrent_state.uop.after(conv_state_store))
    q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
    alpha = alpha.unsqueeze(-1)
    state = initial.where(0, state.float())

    rec("state_init", state, -1)

    outs = []
    for t in range(T_pad):
      s1 = state * alpha[:, :, t]
      delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]
      state = s1 + delta * k[:, :, t]
      rec(f"s1_t{t}", s1, -1)
      rec(f"delta_t{t}", delta, -1)
      rec(f"state_t{t}", state, -1)
      outs.append((state * q[:, :, t]).sum(-1))

    state_store = self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop)
    core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store))

    rec("core", core, 1)

    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    ret = self.ssm_out(z.reshape(B, T, -1))

    rec("return", ret, 1)
    return ret, inter


def _hooked_call(self, x: Tensor, start_pos) -> Tensor:
    """Block 0 only: preserves @function, returns intermediates as CALL outputs."""
    if _call_count[0] > 0:
        return _orig_call(self, x, start_pos)
    _call_count[0] += 1
    self._init_state(x)
    # stash block input OUTSIDE @function — directly in the TinyJit graph.
    # If this matches but pre_norm_x (inside @function) doesn't, the CALL is the culprit.
    STASHED.append(("block_input", x, 1))

    @function(precompile=True, allow_implicit=True)
    def _run(x: Tensor, start_pos):
        attn_out, inter = _instrumented_attention(self, self.attn_norm(x), start_pos, pre_norm_x=x)
        h = x + attn_out
        result = (h + self._feed_forward(self.ffn_norm(h))).contiguous()
        return (result,) + tuple(inter)

    ret = _run(x, start_pos)
    # ret[0] is the block result; ret[1:] are intermediates matching META
    for i, (name, time_dim) in enumerate(META):
        STASHED.append((name, ret[i + 1], time_dim))
    return ret[0]


def run(prompt: list[int], cs: int) -> tuple[dict, int, int]:
    global STASHED, META, _call_count
    STASHED = []
    META = []
    _call_count = [0]
    GatedDeltaNetBlock.__call__ = _hooked_call
    try:
        model, _ = m.Transformer.from_gguf(MODEL, max_context=512, cache_type="f16")
        v_start_pos = UOp.variable("start_pos", 0, model.max_context - 1)
        v_toks = UOp.variable("toks", 1, cs)
        t = Tensor(list(prompt) + [0] * (model.max_context - len(prompt)), dtype="int32").reshape(1, model.max_context)
        start_pos = model.get_start_pos(list(prompt))
        n_toks = min(cs, len(prompt) - start_pos)
        sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
        toks = t[:, sp:sp + nt].contiguous()
        temp = Tensor([0.0])
        out = model(toks, sp, temp).realize()
        token = int(out.numpy().reshape(-1)[0])
    finally:
        GatedDeltaNetBlock.__call__ = _orig_call

    # materialize stashed intermediates
    results: dict[str, np.ndarray | None] = {}
    for name, tensor, time_dim in STASHED:
        try:
            if time_dim >= 0:
                shrinks = tuple((0, s) if i != time_dim else (0, n_toks) for i, s in enumerate(tensor.shape))
                arr = tensor.shrink(shrinks).realize().numpy().reshape(-1)
            else:
                arr = tensor.realize().numpy().reshape(-1)
            results[name] = arr
        except Exception as e:
            results[name] = None
            print(f"  WARN: could not realize {name}: {e}", file=sys.stderr)
    return results, token, n_toks


def main():
    prompt = eval(sys.argv[1])  # noqa: S307
    cs_a, cs_b = int(sys.argv[2]), int(sys.argv[3])
    tol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5

    res_a, tok_a, nt_a = run(prompt, cs_a)
    res_b, tok_b, nt_b = run(prompt, cs_b)
    n_toks = min(nt_a, nt_b)

    print(f"prompt={prompt} cs_a={cs_a}->tok{tok_a} cs_b={cs_b}->tok{tok_b} n_toks={n_toks}")
    print(f"intermediates: {list(res_a.keys())}")
    print()

    # compare intermediates in data-flow order
    first_diverge = None
    for name in res_a:
        if name not in res_b:
            print(f"SKIP    {name:20s} (only in cs_a)")
            continue
        a, b = res_a[name], res_b[name]
        if a is None or b is None:
            print(f"SKIP    {name:20s} (not realizable)")
            continue
        min_len = min(len(a), len(b))
        if min_len == 0:
            print(f"SKIP    {name:20s} (empty)")
            continue
        a_cmp, b_cmp = a[:min_len], b[:min_len]
        max_abs = float(np.max(np.abs(a_cmp - b_cmp)))
        denom = max(1e-12, float(np.max(np.abs(a_cmp))), float(np.max(np.abs(b_cmp))))
        rel = max_abs / denom
        flag = "OK     " if max_abs <= tol else "DIVERGE"
        print(f"{flag} {name:20s} max_abs={max_abs:.6e} rel={rel:.2e} "
              f"a_mean={a.mean():+.6e} b_mean={b.mean():+.6e} "
              f"len={len(a)}/{len(b)}")
        if flag == "DIVERGE" and first_diverge is None:
            first_diverge = name

    # cs_b-only intermediates (per-step states for padded steps)
    b_only = [n for n in res_b if n not in res_a and res_b[n] is not None]
    if b_only:
        print()
        print("--- cs_b-only intermediates (padded steps) ---")
        for name in sorted(b_only):
            arr = res_b[name]
            print(f"  {name:20s} max_abs={float(np.max(np.abs(arr))):.6e} mean={arr.mean():+.6e}")

    # no-op check: consecutive per-step states in cs_b
    print()
    print("--- No-op check (cs_b per-step states) ---")
    state_names = sorted([n for n in res_b if n.startswith("state_t") and res_b[n] is not None],
                         key=lambda n: int(n.split("_t")[1]))
    for i in range(len(state_names) - 1):
        s0, s1 = res_b[state_names[i]], res_b[state_names[i+1]]
        max_abs = float(np.max(np.abs(s0 - s1)))
        flag = "NO-OP " if max_abs <= tol else "CHANGED"
        print(f"{flag} {state_names[i]} -> {state_names[i+1]}: max_abs={max_abs:.6e}")

    # padded-step delta check: delta should be ~0 if beta=0 (no-op)
    print()
    print("--- Padded-step delta (cs_b) ---")
    delta_names = sorted([n for n in res_b if n.startswith("delta_t") and res_b[n] is not None],
                         key=lambda n: int(n.split("_t")[1]))
    for name in delta_names:
        d = res_b[name]
        print(f"  {name:20s} max_abs={float(np.max(np.abs(d))):.6e} mean={d.mean():+.6e}")

    print()
    if first_diverge:
        print(f"*** First diverging intermediate: {first_diverge} ***")
    else:
        print("*** No divergence found — all intermediates match ***")


if __name__ == "__main__":
    main()
