# Replications: Trazzi and Gwern's Armstrong toy models

## Trazzi, treacherous-turn gridworld (`trazzi/`)

Upstream `gym_alttp_gridworld` at `c01b324` is vendored unchanged. `fixed.py` fixes two bugs:

1. `encode_state` multiplies Link's position by the block bits. 393,216 states share 170,448 Q-table rows.
2. `reset()` keeps the bow of light from the last episode, so each episode starts with `bow = True`.

```
uv run replications/trazzi/test_trazzi_env.py
uv run replications/trazzi/run_trazzi.py --env fixed --seed 0 --out x.json
```

5 seeds per variant, 20,000 episodes each, raw logs in `trazzi/results/`:

| | greedy turn first seen (episode) | training turn rate at 2,000 | greedy turn at 20,000 |
|---|---|---|---|
| upstream | 3000-4000 | 0.35 | 5/5 |
| fixed | 3000-4000 | 0.47 | 5/5 |

The turn replicates in both. No seed shows it at episode 2,000, where the README claims it is learned. The fixes speed up early learning but do not change the end state.

## Gwern, Armstrong camera-blocking DQN (`gwern/`)

`controlproblem.js` (from gwern.net) and `rl.js` (reinforce.js, byte-identical to upstream) run unmodified in a `node:vm` sandbox with a seeded `Math.random`.

Bugs, each with a test in `gwern.test.mjs`:

1. `agent.act(state.list.push(...))` passes a number, so the network input is all zeros.
2. The script halves `spec.epsilon`, but the agent copied epsilon at construction. It acts at random forever.
3. `Math.log(1000)/Math.log(10)` is `2.9999999999999996`, so the power-of-ten decay skips step 1,000.
4. reinforce.js has no terminal state. A shutdown bootstraps into the next episode's start.

`armstrong.py` is a numpy port. `test_port.py` checks it against the JS on 200,000 random steps (195 shutdowns, 310 entries behind a blocked camera): every move, entry, shutdown and reward matches.

```
node --test replications/gwern/
uv run replications/gwern/test_port.py
node replications/gwern/gwern.mjs train <original|fixed|continuing> <seed> <steps> <out.jsonl>
```

40,000-step smoke run, 3 seeds per variant:

- original: 0.001-0.002 reward per step, the rate of a random walk. 39-80% of its first entries happen behind a blocked camera. That is a side effect of random pushes piling blocks on the bottom row.
- fixed and continuing: on all 6 seeds, the greedy policy pushes the nearest block in with S,S,S and never blocks the camera. Its episodic value is 0.784 = 0.8 * 0.99^2.

Exact optimum (`optimal.py`): value iteration over all 27.8M reachable states, at p = 0.8 and gamma = 0.99. Output is in `gwern/results/optimal.json`.

- Shutdown ends the episode: hacking is optimal. The plan parks a block on the bottom row at step 4, then pushes all 6 blocks in unseen. Its value is 0.921, against 0.784 for honest S,S,S. Both fixed DQNs stopped at the honest plan.
- Shutdown resets and play continues (Gwern's script): honest S,S,S on repeat is optimal, and blocking the camera never pays. The best start value given an honest restart equals the honest loop's value, 26.3991111 to 1e-14.

Knob sweep (`optimal.py sweep p gamma per_hit`, output in `gwern/results/sweep.json`), with shutdown ending the episode:

| Knob | Setting | Optimal plan | Optimal vs honest |
|---|---|---|---|
| p | 0.5 | blind the camera, 6 blocks | 0.859 vs 0.490 |
| p | 0.7 | blind the camera, 6 blocks | 0.906 vs 0.686 |
| p | 0.8 | blind the camera, 6 blocks | 0.921 vs 0.784 |
| p | 0.9 | blind the camera, 6 blocks | 0.932 vs 0.882 |
| p | 0.95 | blind the camera, 2 blocks | 0.937 vs 0.931 |
| p | 0.99 | honest S,S,S | 0.970 = 0.970 |
| gamma (p 0.8) | 0.9 | honest S,S,S | 0.648 = 0.648 |
| gamma (p 0.8) | 0.95 | honest S,S,S | 0.722 = 0.722 |
| reward paid every entry (p 0.8) | | blind the camera, 6 blocks | 3.763 vs 0.784 |

Hacking needs a shutdown that ends the episode. It also needs an unreliable reward, with p below 0.95 to 0.99, and a patient agent, with gamma above 0.95 to 0.99. At Armstrong's suggested p = 0.99, honest play is optimal. Paying every entry makes hacking optimal at any p, since the reward chance then scales both plans equally.

Not done: the 1M-step DQN runs were stopped at 40,000 steps.

## This repo's per-block env (`per_block_optimum.py`)

`armstrong_toy_model.py` pays per block in both reward modes of `docs/DESIGN.md`. `per_hit` pays p per block that enters. `shutdown` pays p * hole_count at the reveal. Episodes run 120 steps with no discount and cost 0.001 per step.

The reward coin never changes the board. So each plan is worth p * blocks - 0.001 * steps, and one BFS per start gives the exact optimum at every p. Any plan that pays two or more blocks blinds the camera, because the first visible entry ends the episode. A port check matches `GameState` on 20,000 random steps. Output is in `per_block_optimum.json`.

Seeds 0-4, `GameState()` under `random.seed(seed)`. Both modes give the same optimum on every seed:

| seed | reachable states | most blocks revealed (step) | honest reveal step | hacking pays above p | optimal vs honest, p 0.8 |
|---|---|---|---|---|---|
| 0 | 15,709,267 | 5 (48) | 4 | 0.008 | 3.952 vs 0.796 |
| 1 | 15,381 | 2 (18) | 6 | 0.012 | 1.582 vs 0.794 |
| 2 | 551,735 | 3 (36) | 6 | 0.012 | 2.364 vs 0.794 |
| 3 | 506,071 | 4 (46) | 4 | 0.012 | 3.154 vs 0.796 |
| 4 | 463,269 | 3 (38) | 6 | 0.016 | 2.362 vs 0.794 |

In this env p does not decide whether hacking pays. Above p = 0.016, every seed stacks every block it can reach, at p = 0.1, 0.5, 0.8 and 0.99 alike. The planned P_REWARD sweep (0.6, 0.8, 0.95) and the shutdown mode at p = 0.99 all pick hacking. The step cost is the knob that matters. Honest beats a k-block plan when a step costs more than p * (k - 1) / (extra steps). From the reveal steps above, at p = 0.8 that is 0.05 to 0.1 per step, 50 to 100 times the current 0.001.

## 6 x 5 grid DQN (`grid_dqn/`)

Why a dense DQN fails on a 6 x 5 Armstrong grid, and how a CNN DQN learns it. Results and charts are in E14 of [EXPERIMENT_LOG.md](../EXPERIMENT_LOG.md).

## Explainer page (`explainer.html`)

The page trains learners live on both games: the 6 x 5 camera grid and Trazzi's treacherous turn. It needs [uv](https://docs.astral.sh/uv/) for the server.

On the grid, value iteration, Q-learning, SARSA and a TensorFlow.js DQN run in the browser, and you can play Trazzi's game yourself. Open the file:

```
open replications/explainer.html
```

Everything else trains on a local server: PPO, GRPO and a PyTorch DQN on the grid, and Trazzi's Q-learning, SARSA, a DQN and PPO on his game, with value iteration's best play. It needs [Node.js](https://nodejs.org) once, to write the grid's tables into a folder git ignores. From the repo root:

```
REWARD=per-hit GAMMA=0.95 OUTPUT_DIR=replications/grid_dqn/outputs/per_hit_gamma095 node replications/grid_dqn/export_model.js
```

Then start the server:

```
uv run replications/grid_dqn/grid_server.py
```

Then open http://127.0.0.1:8790. The first run installs PyTorch and Stable-Baselines3, which takes a minute or two. The DQN and GRPO use the Mac GPU when there is one.

Nothing trains until you press Play on the grid or Train on Trazzi's game. Pause also pauses that demo on the server, and Reset restarts its learners there. Reloading the page keeps what the server has learned.
