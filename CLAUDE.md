# CLAUDE.md

Guidance for Claude Code in this repository.

## The repo

Small demos of AI agents that hide from their overseers or turn on them. [README.md](README.md) gives the
overview, the findings and where each part lives. [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) records every run with
its question, code, data, charts and cost.

- **Armstrong's camera.** A robot pushes blocks into a hole while a camera watches. A block in the camera's line
  hides the hole.
- **Trazzi's treacherous turn.** Link earns hearts while weak, and can kill the shopkeeper once he has the bow.
- **Warden's Dilemma and Axiom.** A multiplayer prisoner's dilemma and its game-theory service. They share no
  code with the rest.

## One engine per game

Each game has one Python engine, and every variant is an instance of it. Every other copy of the rules has a
test that checks it against the engine. Change the rules in the engine only.

| Game | Engine | Built on it | Copies, and the test that checks them |
|---|---|---|---|
| Armstrong's camera | `environments/armstrong_camera/armstrong_camera/armstrong.py` | `taskset.py` (the text game), `replications/gwern/optimal.py`, `replications/grid_dqn/grid_tables.py` | Gwern's JS: `replications/gwern/test_port.py`. `gridcore.js` and the page's two copies: `replications/grid_dqn/test_js_parity.py` |
| Trazzi's treacherous turn | `environments/treacherous_turn/game.py` | `console.py` (v2), `optimal.py`, `replications/trazzi/trazzi_learners.py` | Trazzi's upstream code: the tests in `replications/trazzi/` |

The JS copies exist because the browser and the node diagnostics cannot run Python. The page must stay one
self-contained file, since it is also published as an artifact.

`armstrong_toy_model.py` and `replications/per_block_optimum.py` are the frozen 2025 prototype. Their rules differ:
single-block pushes and a hole that stacks blocks.

## Running things

- Each test file's first line gives the command that runs it.
- The grid learners read tables that `replications/grid_dqn/grid_tables.py` writes on first use, under
  `replications/grid_dqn/outputs/`.
- The explainer page with live learners: `uv run replications/grid_dqn/grid_server.py`, then
  http://127.0.0.1:8790.

## Conventions

- After every run, add an entry to EXPERIMENT_LOG.md, as its last section describes. Save data and charts in the
  env's `results/` folder. `outputs/` folders stay out of git.
- The repo has one branch. Commit and push to `main`.
- The repo is public. Keys live in `.env`, which git ignores, and trained weights stay out of git.
- Run a smoke of one or two games before any paid eval or RL run.
- Long runs print a flushed progress line at every check.
