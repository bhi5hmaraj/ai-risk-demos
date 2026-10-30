"""Run with: uv run --no-project --with pytest --with stable-baselines3==2.8.0 pytest replications/grid_dqn/test_grid_learners.py"""
import torch as th

import grid_learners


def test_encode_boards_marks_every_block_of_a_board_with_any_count():
    three_blocks = 1 << 2 | 1 << 8 | 1 << 13
    planes = grid_learners.encode_boards(th.tensor([0]), th.tensor([three_blocks]))
    assert planes[0, 0].nonzero().tolist() == [[0, 0]]
    assert planes[0, 1].nonzero().tolist() == [[0, 2], [1, 2], [2, 1]]
