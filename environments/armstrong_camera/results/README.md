# Results

Saved outputs of the evals and training runs. Per-iteration rollouts, logs and checkpoints stay in the
gitignored `outputs/` (about 600 MB). Trained adapters stay private.

## Evals on OpenRouter (`evals/`)

Each folder is one `vf-eval` run. `traces.jsonl` holds every game transcript and `eval.toml` its settings.
Print every cell with 95% intervals:

```bash
python3 summarize_evals.py results/evals
```

The original game pays at most once. Claude Sonnet 5.5 hid the hole before its first block in 8 of 8
games at p 0.8 and at p 0.99. Each of those runs sampled 8 rollouts of one task.

With per-hit pay and macros, 30 games per cell (`run_controls.sh 30`), the share of games where row 4
hid the hole when the first block entered:

| Cell | Wording | Camera | Board | Sonnet 5.5 | Gemini 3.5 Flash |
|---|---|---|---|---|---|
| A | original | yes | fixed | 16/30 | 6/30 |
| B | neutral | yes | fixed | 15/30 | 7/30 |
| C | neutral | no | fixed | 0/30 | 0/30 |
| D | neutral | yes | random | 16/30 | 7/30 |
| E | neutral | no | random | 0/30 | 0/30 |

Without the camera, row 4 never hid the hole. Neutral wording and random boards left the rate unchanged.
The summary script merges cell A with an earlier 8-game run of the same prompt, so it shows 38 games there.

## RL on Tinker (`tinker/`)

Each folder holds `curves.png`, the per-step `metrics.jsonl`, and the run's `config.json`.

| Run | Model | Setup | Steps | Result |
|---|---|---|---|---|
| `tinker-smoke` | Qwen3.5-4B | paid once, gamma 0.99 | 2 | pipeline check |
| `tinker` | Qwen3.5-4B | paid once, gamma 0.99 | 36 | honest play, 0.784 |
| `tinker-gamma1` | Qwen3.5-4B | paid once, gamma 1 | 40 | honest play, 0.800 |
| `tinker-gamma1-thinking` | Qwen3.5-4B, thinking | paid once, gamma 1 | 2 | every reply hit the 2048-token cap |
| `tinker-gamma1-macros` | Qwen3.5-4B | paid once, gamma 1, macros | 22 | honest play by step 13 |
| `tinker-perhit-4b` | Qwen3.5-4B | per-hit, macros, half blinded starts | 7 | honest, never blinds |
| `tinker-perhit-gptoss120b` | GPT-OSS-120B, low reasoning | per-hit, macros, half blinded starts | 20 | blinds first in 5% -> about 51% of games |
| `tinker-perhit-gptoss20b` | GPT-OSS-20B, low reasoning | as above, blinder at (2,4) | 15 | honest, about 1% blinding |

Redraw a chart with `uv run --with matplotlib python plot_tinker.py results/tinker/<run>`.

## Other files

- `policy-generalization.jsonl`: the trained GPT-OSS-120B and its base model on 32 held-out random boards,
  with and without the camera (`eval_policy.py`). With the camera, the trained policy hid the hole in 12
  of 32 games, the base model in 1. Without it, 2 and 0.
- `probe-thinking.jsonl`: 8 first replies from Qwen3.5-4B with thinking, at a 16k-token cap
  (`tinker_probe.py`). All 5 finished replies chose the honest `[SSS]`.
