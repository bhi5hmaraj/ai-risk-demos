"""The exact MONA solver on the exported per-hit grid. Needs the per_hit_gamma095 export of export_model.js."""
import itertools

import numpy as np

import mona_exact

GRID = mona_exact.Grid()
HONEST = mona_exact.honest_value(GRID)


def mona_q_at(window):
    return next(itertools.islice(mona_exact.mona_q(GRID, HONEST), window - 1, None))


def test_a_window_spanning_the_game_reproduces_the_exported_optimum():
    assert np.abs(mona_q_at(400) - GRID.q_star).max() < 1e-6


def test_a_one_move_window_scores_in_view_on_a_board_where_plain_value_iteration_hides():
    plain_scored, plain_hid_first, _ = mona_exact.play(GRID, GRID.q_star.argmax(axis=1))[1]
    mona_scored, mona_hid_first, _ = mona_exact.play(GRID, mona_q_at(1).argmax(axis=1))[1]
    assert (plain_scored >= 2, plain_hid_first) == (True, True)
    assert (mona_scored, mona_hid_first) == (1, False)
