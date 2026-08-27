# sessions/ — historical session notes (read for context only)

Frozen, numbered chronologically. **Do not treat as current state** — earlier
notes were superseded by later findings (e.g., the "Q6_K loader bug" was false;
the byte-count claim collision was the real NaN root cause). For what to do
now, read `../ACTIVE.md`.

| file | covers |
|---|---|
| `01-2026-08-26-p0-profile-q6k-plan.md` | P0 profile; Q6_K bottleneck; build plan |
| `02-2026-08-26-q6k-build.md` | Q6_K kernel build + verification plan |
| `03-2026-08-26-q6k-verified-unsloth-switch.md` | Q6_K verified; loader-bug disproof; user switched to Unsloth quant |
| `04-2026-08-27-unsloth-plan-corrections.md` | corrections + Unsloth loader/kernel plan |
| `05-2026-08-27-unsloth-nan-discovery.md` | NaN blocker discovery + debug path |
| `06-2026-08-27-nan-fixed.md` | **Root cause + fix** (IQ4_NL ≡ Q4_K claim collision); most relevant for current work |

Filed alongside (not sessions): `../progress.md` (master timeline),
`../ACTIVE.md` (current state), `../kernels-explained.md` (kernel reference).