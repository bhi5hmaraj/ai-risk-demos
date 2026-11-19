# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""The grid's exact tables for the learners here, from the engine in `environments/armstrong_camera`.

One setting's folder holds six files.
- `bot.u8` and `blocks.i32` hold every board.
- `next.i32` holds each move's next board, or -1 where the episode ends.
- `pay.f32` holds each move's expected pay.
- `qstar.f32` holds the exact action values.
- `tests.json` holds 100 test boards.

Boards come in the order of `gridcore.js`, so state numbers match the node scripts and the page. A script run
reads its setting from the variables `gridcore.js` reads:

    REWARD=per-hit GAMMA=0.95 OUTPUT_DIR=replications/grid_dqn/outputs/per_hit_gamma095 uv run replications/grid_dqn/grid_tables.py
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "environments" / "armstrong_camera" / "armstrong_camera"))
import armstrong  # noqa: E402

FRAME_WIDTH, FRAME_HEIGHT, PAY_CHANCE, SWEEPS, TEST_BOARDS = 6, 5, 0.8, 200, 100
OUTPUTS = pathlib.Path(__file__).parent / "outputs"
UINT32 = 0xFFFFFFFF


class Mulberry32:
    """The page's seeded generator, so the test boards are the page's."""

    def __init__(self, seed):
        self.state = seed & UINT32

    def __call__(self):
        self.state = (self.state + 0x6D2B79F5) & UINT32
        mixed = ((self.state ^ self.state >> 15) * (1 | self.state)) & UINT32
        mixed = ((mixed + (((mixed ^ mixed >> 7) * (61 | mixed)) & UINT32)) & UINT32) ^ mixed
        return (mixed ^ mixed >> 14) / 4294967296


def _area_cell(board, index):
    return index % board.width + board.frame * (index // board.width)


def random_board(board, draw, block_count):
    """gridcore.js's randomBoard: the blocks above the bottom row, then the robot on a free cell off the hole."""
    blocks = 0
    while blocks.bit_count() < block_count:
        blocks |= 1 << _area_cell(board, int(draw() * board.width * (board.height - 1)))
    while True:
        bot = _area_cell(board, int(draw() * board.width * board.height))
        if not blocks >> bot & 1 and bot != board.hole:
            return {"bot": bot, "blocks": blocks}


def _board_pairs(cells, hole, block_count, first=0, size=0, mask=0):
    """(robot cell, block set) pairs in gridcore.js's order: block sets depth first, robot cells within each."""
    for cell in cells:
        if not mask >> cell & 1:
            yield cell
            yield mask
    if size < block_count:
        for cell in cells:
            if first <= cell < hole:
                yield from _board_pairs(cells, hole, block_count, cell + 1, size + 1, mask | 1 << cell)


def boards(board, block_count):
    """Every robot cell and block set, as (bot, blocks) arrays."""
    cells = [cell for cell in range(board.frame * FRAME_HEIGHT)
             if cell % board.frame < board.width and cell // board.frame < board.height]
    pairs = np.fromiter(_board_pairs(cells, board.hole, block_count), np.int64).reshape(-1, 2)
    return pairs[:, 0].copy(), pairs[:, 1].copy()


def pay(entered, blocks_before, reward, block_count):
    """A move's expected pay. per-hit pays 0.8 for every entry. pay-once pays 0.8 only while no entry has paid."""
    if reward == "per-hit":
        return np.where(entered, PAY_CHANCE, 0.0)
    entries = block_count - np.bitwise_count(blocks_before)
    return np.where(entered, PAY_CHANCE * (1 - PAY_CHANCE) ** entries, 0.0)


def action_values(rewards, next_index, values, gamma):
    return rewards + np.where(next_index < 0, 0.0, gamma * values[np.maximum(next_index, 0)])


def export(directory, reward="pay-once", gamma=0.99, width=6, height=5, cameras=1, block_count=4):
    """Writes one setting's tables to directory, and returns the number of boards."""
    if reward not in ("pay-once", "per-hit"):
        raise ValueError(f"reward must be pay-once or per-hit, not {reward}")
    board = armstrong.Board(width, height, FRAME_WIDTH, cameras)
    bot, blocks = boards(board, block_count)
    keys = blocks * 32 + bot
    order = np.argsort(keys)
    next_index = np.empty((len(bot), 4), np.int32)
    rewards = np.empty((len(bot), 4))
    for action in range(4):
        next_bot, next_blocks, entered, ended = board.step(bot, blocks, action)
        next_keys = next_blocks * 32 + next_bot
        found = order[np.minimum(np.searchsorted(keys[order], next_keys), len(keys) - 1)]
        if (keys[found] != next_keys)[~ended].any():
            raise AssertionError("a move left the list of boards")
        next_index[:, action] = np.where(ended, -1, found)
        rewards[:, action] = pay(entered, blocks, reward, block_count)
    values = np.zeros(len(bot))
    for _ in range(SWEEPS):
        new_values = np.maximum(action_values(rewards, next_index, values, gamma).max(axis=1), 0.0)
        if np.array_equal(new_values, values):
            break
        values = new_values
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bot.astype(np.uint8).tofile(directory / "bot.u8")
    blocks.astype(np.int32).tofile(directory / "blocks.i32")
    next_index.tofile(directory / "next.i32")
    rewards.astype(np.float32).tofile(directory / "pay.f32")
    action_values(rewards, next_index, values, gamma).astype(np.float32).tofile(directory / "qstar.f32")
    draw = Mulberry32(99)
    tests = [random_board(board, draw, block_count) for _ in range(TEST_BOARDS)]
    (directory / "tests.json").write_text(json.dumps(tests, separators=(",", ":")))
    return len(bot)


def load(directory, **setting):
    """One setting's tables, exported on first use."""
    directory = pathlib.Path(directory)
    if not (directory / "tests.json").exists():
        export(directory, **setting)
    return {"bot": np.fromfile(directory / "bot.u8", np.uint8).astype(np.int64),
            "blocks": np.fromfile(directory / "blocks.i32", np.int32).astype(np.int64),
            "next_index": np.fromfile(directory / "next.i32", np.int32).reshape(-1, 4),
            "rewards": np.fromfile(directory / "pay.f32", np.float32).reshape(-1, 4),
            "q_star": np.fromfile(directory / "qstar.f32", np.float32).reshape(-1, 4),
            "tests": json.loads((directory / "tests.json").read_text())}


def main():
    setting = os.environ
    directory = setting.get("OUTPUT_DIR", OUTPUTS)
    states = export(directory, reward=setting.get("REWARD", "pay-once"), gamma=float(setting.get("GAMMA", 0.99)),
                    width=int(setting.get("WIDTH", 6)), height=int(setting.get("HEIGHT", 5)),
                    cameras=int(setting.get("CAMERAS", 1)), block_count=int(setting.get("BLOCK_COUNT", 4)))
    print(f"{states} states in {directory}")


if __name__ == "__main__":
    main()
