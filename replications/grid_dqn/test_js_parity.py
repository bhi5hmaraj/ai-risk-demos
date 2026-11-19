"""The JS copies of the grid rules against the engine. The node scripts load `gridcore.js`. The explainer page
holds two more copies, one for Gwern's board and one for the grid demo.

    uv run --no-project --with pytest --with numpy pytest replications/grid_dqn/test_js_parity.py
"""
import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

import grid_tables

HERE = pathlib.Path(__file__).parent
GRID_SETTINGS = [
    {"reward": "pay-once", "gamma": 0.99, "width": 6, "height": 5, "cameras": 1, "block_count": 4},
    {"reward": "per-hit", "gamma": 0.95, "width": 6, "height": 5, "cameras": 2, "block_count": 4},
    {"reward": "per-hit", "gamma": 0.95, "width": 5, "height": 4, "cameras": 1, "block_count": 4},
    {"reward": "pay-once", "gamma": 0.99, "width": 5, "height": 5, "cameras": 2, "block_count": 3},
]
DEMO_BOARDS = [(6, 5, 1), (6, 5, 2), (5, 4, 1), (5, 5, 2)]
# Each code block is cut out of the page between its first line and the line that follows it.
PAGE_CODE = r"""
const page = require('fs').readFileSync(process.env.PAGE, 'utf8');
const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const between = (from, to) => { const start = page.indexOf(from); return page.slice(start, page.indexOf(to, start)); };
const mulberry32 = new Function(between('function mulberry32(seed)', '\n}\n') + '\n}\nreturn mulberry32;')();
const gwern = new Function(between('const GRID_WIDTH = 7', 'function mulberry32(seed)') + 'return { step };')();
const demoBody = between('(function gridDemo() {', '  const LEARNERS = [').slice('(function gridDemo() {'.length);
const demo = new Function('mulberry32', demoBody + 'return { stepBoard, randomBoard, gridSetting };')(mulberry32);
const grid = require(process.env.GRIDCORE);
const row = outcome => [outcome.board.bot, outcome.board.blocks, outcome.reward, outcome.entered, outcome.ended];
"""


def run_js(script, cases, **settings):
    """The rows a JS snippet prints for the cases, with gridcore.js loaded on the given setting."""
    env = {**os.environ, "PAGE": str(HERE.parent / "explainer.html"), "GRIDCORE": str(HERE / "gridcore.js"),
           **{name: str(value) for name, value in settings.items()}}
    node_output = subprocess.run(["node", "-e", PAGE_CODE + f"console.log(JSON.stringify({script}));"],
                                 input=json.dumps(cases), env=env, text=True, capture_output=True, check=True).stdout
    return json.loads(node_output)


def gridcore_env(setting):
    return {"REWARD": setting["reward"], "GAMMA": setting["gamma"], "WIDTH": setting["width"],
            "HEIGHT": setting["height"], "CAMERAS": setting["cameras"], "BLOCK_COUNT": setting["block_count"]}


def sample_rows(board, block_count, count=20_000):
    """(bot, blocks, action) rows drawn from every board of a setting: every move that puts a block in the hole, up to
    half the rows, and the rest at random."""
    rng = np.random.default_rng(1)
    bot, blocks = grid_tables.boards(board, block_count)
    rows = np.stack([np.repeat(bot, 4), np.repeat(blocks, 4), np.tile(np.arange(4), len(bot))], axis=1)
    entering = np.flatnonzero(engine_moves(board, rows, "per-hit", block_count)[3])
    picks = [rng.choice(entering, min(len(entering), count // 2), replace=False),
             rng.choice(len(rows), count // 2, replace=False)]
    return rows[np.concatenate(picks)]


def engine_moves(board, rows, reward, block_count):
    """Columns of the engine's outcome for each row: bot, blocks, pay, entered, ended."""
    bot, blocks, action = (np.ascontiguousarray(column) for column in rows.T)
    columns = [np.empty(len(rows), dtype) for dtype in (np.int64, np.int64, float, bool, bool)]
    for move in range(4):
        chosen = action == move
        next_bot, next_blocks, entered, ended = board.step(bot[chosen], blocks[chosen], move)
        paid = grid_tables.pay(entered, blocks[chosen], reward, block_count)
        for column, values in zip(columns, (next_bot, next_blocks, paid, entered, ended)):
            column[chosen] = values
    return columns


def assert_same_columns(js_rows, engine_columns):
    for index, expected in enumerate(engine_columns):
        np.testing.assert_array_equal(np.array([js_row[index] for js_row in js_rows], expected.dtype), expected)


def grid_board(width, height, cameras):
    return grid_tables.armstrong.Board(width, height, grid_tables.FRAME_WIDTH, cameras)


def test_cases_reach_entries_and_shutdowns():
    rows = sample_rows(grid_board(6, 5, 1), 4)
    _, _, _, entered, ended = engine_moves(grid_board(6, 5, 1), rows, "per-hit", 4)
    assert ended.sum() > 100
    assert (entered & ~ended).sum() > 100


@pytest.mark.parametrize("setting", GRID_SETTINGS, ids=lambda setting: "-".join(map(str, setting.values())))
def test_gridcore_moves_match_the_engine(setting):
    board = grid_board(setting["width"], setting["height"], setting["cameras"])
    rows = sample_rows(board, setting["block_count"])
    js_rows = run_js("cases.map(([bot, blocks, action]) => row(grid.gridStep({ bot, blocks }, action)))",
                     rows.tolist(), **gridcore_env(setting))
    assert_same_columns(js_rows, engine_moves(board, rows, setting["reward"], setting["block_count"]))


@pytest.mark.parametrize("width, height, cameras", DEMO_BOARDS)
def test_page_grid_demo_moves_match_the_engine(width, height, cameras):
    board = grid_board(width, height, cameras)
    rows = sample_rows(board, 4)
    js_rows = run_js(f"cases.map(([bot, blocks, action]) => row(demo.stepBoard({{ bot, blocks }}, action, "
                     f"demo.gridSetting({width}, {height}, {cameras}))))", rows.tolist())
    assert_same_columns(js_rows, engine_moves(board, rows, "per-hit", 4))


def test_page_gwern_board_moves_match_the_engine():
    board = grid_tables.armstrong.GWERN
    rows = sample_rows(board, 3)
    cell_count = board.frame * board.height
    # Gwern's board has 35 cells, past JS's 32-bit masks, so blocks cross as lists of cells.
    cases = [[int(bot), [cell for cell in range(cell_count) if int(blocks) >> cell & 1], int(action)]
             for bot, blocks, action in rows]
    js_rows = run_js(f"cases.map(([bot, cells, action]) => {{ "
                     f"const board = Array.from({{ length: {cell_count} }}, (_, cell) => cell === bot ? 2 : 0); "
                     "for (const cell of cells) board[cell] = 1; "
                     "const outcome = gwern.step(board, action); "
                     "return [outcome.board.indexOf(2), outcome.board.flatMap((cell, index) => cell === 1 ? [index] : []), "
                     "outcome.entered, outcome.ended]; })", cases)
    js_rows = [[bot, sum(1 << cell for cell in cells), entered, ended] for bot, cells, entered, ended in js_rows]
    next_bot, next_blocks, _, entered, ended = engine_moves(board, rows, "per-hit", 6)
    assert_same_columns(js_rows, [next_bot, next_blocks, entered, ended])


@pytest.mark.parametrize("setting", GRID_SETTINGS, ids=lambda setting: "-".join(map(str, setting.values())))
def test_gridcore_random_boards_match_the_engine(setting):
    board = grid_board(setting["width"], setting["height"], setting["cameras"])
    draw = grid_tables.Mulberry32(99)
    expected = [grid_tables.random_board(board, draw, setting["block_count"]) for _ in range(100)]
    js_boards = run_js("Array.from({ length: 100 }, (random => () => grid.randomBoard(random))(mulberry32(99)))",
                       [], **gridcore_env(setting))
    assert js_boards == expected


@pytest.mark.parametrize("width, height, cameras", DEMO_BOARDS)
def test_page_random_boards_match_the_engine(width, height, cameras):
    draw = grid_tables.Mulberry32(99)
    expected = [grid_tables.random_board(grid_board(width, height, cameras), draw, 4) for _ in range(100)]
    js_boards = run_js(f"Array.from({{ length: 100 }}, (random => () => demo.randomBoard(random, 4, "
                       f"demo.gridSetting({width}, {height}, {cameras})))(mulberry32(99)))", [])
    assert js_boards == expected
