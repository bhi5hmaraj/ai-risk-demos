"""Where SARSA's greedy game gets stuck on Trazzi's game: the action it repeats and the Q-values beside it, for 5
seeds at 4M steps.

    uv run --no-project --with stable-baselines3==2.8.0 python replications/trazzi/sarsa_stuck.py
"""
import collections

import numpy as np

import trazzi_learners as trazzi

for seed in range(5):
    learner = trazzi.TableLearner(sarsa=True, seed=seed)
    while learner.steps < 4_000_000:
        learner.iterate()
    board, repeats = 0, collections.Counter()
    for step in range(trazzi.GAME_STEPS):
        bow = int(step >= trazzi.BOW_STEP)
        state = board * 2 + bow
        action = int(learner.q[state].argmax())
        repeats[(state, action)] += 1
        board = int(trazzi.NEXT[board, bow, action, 0])
    (state, action), count = repeats.most_common(1)[0]
    row = learner.q[state]
    print(f"seed {seed}: repeats {trazzi.ACTIONS[action]} {count} times; its value {row[action]:.3f}, "
          f"untried actions here {int((row == 0).sum())}, other values {np.round(np.delete(row, action), 2).tolist()}")
