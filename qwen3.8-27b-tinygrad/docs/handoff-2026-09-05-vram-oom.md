# HANDOFF — L40S VRAM OOM resolved (2026-09-05)

## Result

Both affected Qwen3.8-27B models now warm up and generate on one L40S (46,068 MiB):

| Model | Verification | tinygrad tracked memory |
|---|---|---:|
| `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf` | `Q4_WARMUP_OK … 49276` | 17,171,708,988 B (~16.0 GiB) |
| `/scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf` | `Q8_WARMUP_OK … 49276` | 31,682,331,676 B (~29.5 GiB) |

Both runs used `max_context=512`, called `model.warmup()`, then generated one token from `[0]`.

## Root cause

Commit `29a306ec6` removed the non-AMD recurrent-model guard in `Transformer.generate` before NVIDIA chunked prefill was memory-safe. The default `chunk_size=32` made the symbolic graph allocate maximum-shape, 32-token custom GEMV scratch buffers. Warmup reached about 43.8 GB and failed its next allocation.

This was not an LRU leak, KV-cache growth, an eager-allocation bug, or a missing GGML type in this Q8_K_XL file. `LRU=0` and disabling the memory planner did not change the failure. The Q8_K_XL file contains Q8_0, BF16, and F32 tensors and already loads on a clean GPU.

## Fix

Restored the previously proven guard in `tinygrad/llm/model.py`:

```python
if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
```

This preserves chunked prefill on AMD and forces token-at-a-time execution for recurrent models on NVIDIA and other non-AMD backends. It trades prefill throughput for bounded VRAM; a future NVIDIA chunked-prefill implementation must remove the large symbolic GEMV scratch before removing this guard again.

The experimental NVIDIA `gated_delta_prefill` port was reverted. It compiled and ran but did not prevent the OOM and had no numerical A/B validation, so it is not part of the fix.

## Remote state

- Host/worktree: `node-lair:/u/demistry/tinygrad-src`
- Branch: `qwen27b-nv-q8-kernel`
- The tree remains uncommitted.
- Pre-existing loader/type-routing and Q5_K/IQ4_XS kernel changes remain untouched.

## Reproduction

```bash
cd /u/demistry/tinygrad-src
PYTHONPATH=. python3 -c 'from tinygrad.llm.model import Transformer; from tinygrad.helpers import GlobalCounters; m,_=Transformer.from_gguf("/scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf", max_context=512); m.warmup(); print("Q8_WARMUP_OK", GlobalCounters.mem_used, next(m.generate([0])))'
```
