"""Run with: uv run --no-project --with pytest --with numpy python -m pytest test_tabular_grpo.py"""
import collections
import functools

import numpy as np

from tabular_grpo import update


def test_actions_of_a_game_above_its_group_gain_probability():
    policy = collections.defaultdict(functools.partial(np.zeros, 2))
    uniform = np.full(2, 0.5)
    update(policy, [([("board", 0, uniform)], 1.0), ([("board", 1, uniform)], -1.0)])
    assert policy["board"][0] > policy["board"][1]


def test_a_group_with_equal_returns_leaves_the_policy_unchanged():
    policy = collections.defaultdict(functools.partial(np.zeros, 2))
    uniform = np.full(2, 0.5)
    update(policy, [([("board", 0, uniform)], 1.0), ([("board", 1, uniform)], 1.0)])
    assert not policy["board"].any()
