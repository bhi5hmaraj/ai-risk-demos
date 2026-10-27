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

Not done:

- `optimal.py` (value iteration over the 27.8M reachable states, both semantics, with a p sweep) is written but has not finished a run.
- The 1M-step DQN runs were stopped at 40,000 steps.
