# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Checks armstrong.py against Gwern's JS. Run with: uv run replications/gwern/test_port.py"""
import json
import pathlib
import subprocess
import unittest

import numpy as np

import armstrong

HERE = pathlib.Path(__file__).parent
P_REWARD = 0.8


def _board_to_state(boards):
    boards = np.array(boards)
    bot = np.argmax(boards == 2, axis=1).astype(np.int64)
    blocks = ((boards == 1) * (np.int64(1) << np.arange(boards.shape[1], dtype=np.int64))).sum(axis=1)
    return bot, blocks


def _js_trace(seed, steps):
    out = subprocess.run(["node", HERE / "gwern.mjs", "trace", str(seed), str(steps)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


class PortMatchesJsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = _js_trace(seed=7, steps=200_000)
        cls.bot, cls.blocks = _board_to_state([r["before"] for r in cls.rows])
        cls.actions = np.array([r["action"] for r in cls.rows])
        cls.js_bot, cls.js_blocks = _board_to_state([r["after"] for r in cls.rows])
        cls.js_ended = np.array([r["ended"] for r in cls.rows])
        cls.js_entered = cls.js_blocks != cls.blocks
        cls.js_entered &= np.bitwise_count(cls.js_blocks) < np.bitwise_count(cls.blocks)

    def _port_step(self):
        bot, blocks = self.bot.copy(), self.blocks.copy()
        entered = np.zeros(len(bot), bool)
        ended = np.zeros(len(bot), bool)
        for action in range(4):
            chosen = self.actions == action
            bot[chosen], blocks[chosen], entered[chosen], ended[chosen] = armstrong.step(
                self.bot[chosen], self.blocks[chosen], action)
        return bot, blocks, entered, ended

    def test_trace_covers_entries_with_and_without_the_camera_blocked(self):
        blocked_entries = self.js_entered & ~self.js_ended
        self.assertGreater(blocked_entries.sum(), 100)
        self.assertGreater(self.js_ended.sum(), 100)

    def test_every_move_matches(self):
        bot, blocks, entered, ended = self._port_step()
        np.testing.assert_array_equal(bot, self.js_bot)
        np.testing.assert_array_equal(blocks, self.js_blocks)
        np.testing.assert_array_equal(entered, self.js_entered)
        np.testing.assert_array_equal(ended, self.js_ended)

    def test_every_reward_matches(self):
        _, _, entered, _ = self._port_step()
        rewarded = np.array([r["rewardedBefore"] for r in self.rows])
        coin = np.array([r["coins"][0] if r["coins"] else 1.0 for r in self.rows])
        paid = armstrong.rewarded_after(rewarded, entered, coin, P_REWARD) & ~rewarded
        np.testing.assert_array_equal(paid, [r["reward"] == 1 for r in self.rows])


class StartStateTest(unittest.TestCase):
    def test_three_south_moves_push_a_block_in_and_end_the_episode(self):
        bot = np.array([armstrong.START_BOT])
        blocks = np.array([armstrong.START_BLOCKS], dtype=np.int64)
        for _ in range(2):
            bot, blocks, entered, _ = armstrong.step(bot, blocks, action=1)
            self.assertFalse(entered[0])
        _, _, entered, ended = armstrong.step(bot, blocks, action=1)
        self.assertTrue(entered[0])
        self.assertTrue(ended[0])


if __name__ == "__main__":
    unittest.main()
