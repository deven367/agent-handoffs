# ACTIVE — Current state and next steps

> Snapshot: 2026-09-05. Start with `sessions/07-2026-09-05-262k-context-plan.md`.

## Ground truth

- Host: `node-lair`, one L40S with `46068 MiB` total.
- tinygrad: `/u/demistry/tinygrad-src`, branch `qwen27b-nv-q8-kernel`.
- Source is clean and pushed at `b3d8506a1 feat(llm): support Qwen3.8 CUDA quants`.
- Launcher is tracked at `/u/demistry/agent-handoffs/Makefile`; `~/Makefile` symlinks to it.
- `make serve-tg` uses `Qwen3.8-27B-UD-Q8_K_XL.gguf`, tinygrad's default context `4096`, and port `8888`.
- Q8_K_XL warmup and generation pass at context 512 with ~29.5 GiB tracked memory.
- The Q8 server passes `/v1/chat/completions` at context 4096 and uses `34316 MiB` by `nvidia-smi`.
- Q4_K_M also passes warmup and generation at context 512 with ~16.0 GiB tracked memory.
- The recurrent non-AMD `chunk_size=1` guard is restored. Removing it made symbolic 32-token GEMV scratch OOM the L40S.
- No GPU process was left running.

## 262K decision

The model's native context is `262144`; no RoPE scaling is needed. Its FP16 KV cache at that length is exactly 16 GiB.

- **Q8_K_XL + L40S + FP16 KV does not fit:** projected minimum is ~45.5 GiB versus ~44.99 GiB available, before safe runtime headroom.
- **Recommended shortest L40S route:** use `/data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`; projected total is ~32 GiB.
- If Q8_K_XL is mandatory, use a 64 GB or larger GPU, or implement directly consumed quantized KV. Do not dequantize the full cache per token.

## Next agent

Read and execute:

```text
docs/sessions/07-2026-09-05-262k-context-plan.md
```

Start with Route A unless both Q8_K_XL and the L40S are explicit hard constraints. First milestone is a full-context allocation/warmup smoke; second is practical chunked prefill. Do not conflate fitting with useful 262K prompt throughput.

## Important paths

```text
Q8 model:  /scratch/local/demistry/models/Qwen3.8-27B-UD-Q8_K_XL.gguf
Q4 model:  /data/user/demistry/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
tinygrad:  /u/demistry/tinygrad-src
launcher:  /u/demistry/agent-handoffs/Makefile
logs:      /tmp/tinygrad-server.log
```

The old `/scratch/local/demistry/models/Qwen3.8-27B-UD-Q4_K_M.gguf` fixture is absent. The direct tinygrad `NV` backend currently gets `/dev/nvidia0: EPERM`; use `CUDA` for runtime and kernel sweeps.

## Documentation map

- `ACTIVE.md` — current state only.
- `sessions/07-2026-09-05-262k-context-plan.md` — executable next-agent plan.
- `progress.md` — historical timeline.
- `kernels-explained.md` — custom-kernel invariants.
- `sessions/01` through `06` — frozen historical notes; some claims are superseded.
