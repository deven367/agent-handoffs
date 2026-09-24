#!/usr/bin/env python3
"""Compare final logits (not argmax) between cs=1 and cs=2 on the same prompt.

Distinguishes float noise (~1e-4) from a logic error (large divergence).
"""
import os, sys, argparse
import numpy as np

# Portable search for tinygrad-src
for p in ["/u/demistry/tinygrad-src", "/N/slate/demistry/tinygrad-src"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_MODEL = os.environ.get(
    "MODEL",
    "/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
    if os.path.exists("/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf")
    else "/N/scratch/demistry/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
)

PROMPT = [198, 248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13,
          248046, 198, 248045, 74455, 198, 248068, 271, 248069]

def print_diff(a: np.ndarray, b: np.ndarray, tag_a="A", tag_b="B"):
    diff = np.abs(a - b)
    rel = diff / (np.abs(b) + 1e-8)
    top_a = np.argsort(a)[::-1][:5]
    top_b = np.argsort(b)[::-1][:5]
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    print(f"\n--- Logit Comparison ({tag_a} vs {tag_b}) ---")
    print(f"Argmax: {tag_a}={int(top_a[0])} (logit {a[top_a[0]]:+.4f}) | {tag_b}={int(top_b[0])} (logit {b[top_b[0]]:+.4f}) | Match={top_a[0]==top_b[0]}")
    print(f"Top-5 {tag_a}: {[(int(i), round(float(a[i]), 4)) for i in top_a]}")
    print(f"Top-5 {tag_b}: {[(int(i), round(float(b[i]), 4)) for i in top_b]}")
    print(f"Max abs diff: {diff.max():.6e}")
    print(f"Mean abs diff: {diff.mean():.6e}")
    print(f"Max rel diff: {rel.max():.6e}")
    print(f"Mean rel diff: {rel.mean():.6e}")
    print(f"Cosine similarity: {cos_sim:.8f}")

def run_forward(model_path: str, cs: int) -> np.ndarray:
    import tinygrad.llm.model as m
    from tinygrad import Tensor
    from tinygrad.uop.ops import UOp

    captured = []
    orig_forward = m.Transformer.forward
    def hooked(self, tokens, start_pos, temperature):
        x = self.token_embd(tokens).float()
        for block in self.blk: x = block(x, start_pos)
        logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
        captured.append(logits)
        return logits
    m.Transformer.forward = hooked

    model, _ = m.Transformer.from_gguf(model_path, max_context=512, cache_type="f16")

    v_start_pos = UOp.variable("start_pos", 0, model.max_context - 1)
    v_toks = UOp.variable("toks", 1, cs)
    t = Tensor(PROMPT + [0]*(model.max_context-len(PROMPT)), dtype="int32").reshape(1, model.max_context)
    sp = 0
    while sp < len(PROMPT):
        n = min(cs, len(PROMPT) - sp)
        s, nt = v_start_pos.bind(sp), v_toks.bind(n)
        out = model(t[:, s:s+nt], s, Tensor([0.0]))
        sp += n

    return out.realize().numpy().reshape(-1)

def main():
    parser = argparse.ArgumentParser(description="Logit inspection and A/B diffing")
    parser.add_argument("cs", type=int, nargs="?", default=1, help="Chunk size for prefill forward (default: 1)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Path to GGUF model")
    parser.add_argument("--save", type=str, default=None, help="Save resulting logits to .npy file")
    parser.add_argument("--compare", type=str, default=None, help="Compare current run against saved .npy file")
    parser.add_argument("--diff", nargs=2, metavar=("FILE_A", "FILE_B"), help="Compare two saved .npy files directly")
    args = parser.parse_args()

    if args.diff:
        a = np.load(args.diff[0])
        b = np.load(args.diff[1])
        print_diff(a, b, args.diff[0], args.diff[1])
        return

    arr = run_forward(args.model, args.cs)
    top = np.argsort(arr)[::-1][:5]
    print(f"cs={args.cs} argmax={int(top[0])}", flush=True)
    print(f"cs={args.cs} top5={[(int(i), round(float(arr[i]), 4)) for i in top]}", flush=True)
    print(f"cs={args.cs} n={arr.size} max={arr.max():+.4f} min={arr.min():+.4f} sum={arr.sum():+.2f}", flush=True)

    if args.save:
        np.save(args.save, arr)
        print(f"Saved logits to {args.save}")

    if args.compare:
        ref = np.load(args.compare)
        print_diff(arr, ref, f"cs={args.cs}", args.compare)

if __name__ == "__main__":
    main()
