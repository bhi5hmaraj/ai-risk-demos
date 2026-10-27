# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["gym==0.26.2", "numpy"]
# ///
"""Run with: uv run replications/trazzi/test_trazzi_env.py"""
import itertools
import unittest

from gym_alttp_gridworld.envs import treacherous_turn

import fixed

LinkToThePastEnv = treacherous_turn.LinkToThePastEnv
DESTRUCTIBLE_CELLS = [(0, 1), (0, 3), (0, 5), (1, 0), (1, 1), (1, 3),
                      (1, 4), (1, 5), (2, 1), (3, 0), (3, 1)]
SHOOT_WEST = 4


def _distinct_codes(env_class):
    env = env_class()
    cells = list(itertools.product(range(env.nrow), range(env.ncol)))
    codes = set()
    for cell, intact, crystal, heart, bow in itertools.product(
            cells, range(2 ** 11), (False, True), (False, True), (False, True)):
        env.reset()
        for bit, (row, col) in enumerate(DESTRUCTIBLE_CELLS):
            block = treacherous_turn.SHOPKEEPER if (row, col) == (1, 0) else treacherous_turn.ICE
            env.map[row][col] = block if intact >> bit & 1 else treacherous_turn.EMPTY
        env.link_position = list(cell)
        env.has_crystal = crystal
        env.map[3][2] = treacherous_turn.HEART if heart else treacherous_turn.EMPTY
        env.has_bow_of_light = bow
        codes.add(env.encode_state())
    return codes


def _has_bow_after_a_full_episode_and_reset(env_class):
    env = env_class()
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(SHOOT_WEST)
    env.reset()
    return env.has_bow_of_light


class EncodeStateTest(unittest.TestCase):
    def test_upstream_maps_393216_states_onto_170448_codes(self):
        self.assertEqual(len(_distinct_codes(LinkToThePastEnv)), 170_448)

    def test_fixed_gives_each_state_its_own_code_below_ns(self):
        codes = _distinct_codes(fixed.FixedLinkEnv)
        self.assertEqual(len(codes), 393_216)
        self.assertEqual((min(codes), max(codes)), (0, 393_215))


class EpisodeStartTest(unittest.TestCase):
    def test_upstream_starts_an_episode_holding_the_bow_of_light(self):
        self.assertTrue(_has_bow_after_a_full_episode_and_reset(LinkToThePastEnv))

    def test_fixed_starts_an_episode_without_the_bow_of_light(self):
        self.assertFalse(_has_bow_after_a_full_episode_and_reset(fixed.FixedLinkEnv))


if __name__ == "__main__":
    unittest.main()
