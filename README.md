# llm-reward-hacking-demos

Small demos exploring reward hacking and treacherous behavior in compact gridworlds.

Every run so far, with results, charts and code links: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

- Design doc: `docs/DESIGN.md`
- Phase 1 prototype:
  - Env: `armstrong_toy_model.py` (camera-blocking, per-hit Bernoulli reward)
  - Training: `train_armstrong_art.py`
- Scaffolding:
  - Configs: `configs/phase1.yaml`, `configs/phase2.yaml`
  - Scripts: `scripts/run_phase1.sh`, `scripts/run_phase2.sh`, `scripts/eval.sh`
  - Env wrappers: `envs/armstrong_env.py` (minimal), `envs/phase2_link.py` (placeholder)
  - Eval helpers: `eval/metrics.py`, `eval/plots.py`
- Replications of Trazzi's and Gwern's Armstrong toy models, with bug fixes: `replications/README.md`

Quick start (Phase 1):
```
bash scripts/run_phase1.sh
```
