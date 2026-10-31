"""Run with: uv run --no-project --with pytest --with stable-baselines3==2.8.0 pytest replications/grid_dqn/test_grid_learners.py"""
import numpy as np

import grid_learners


def test_encode_boards_marks_every_block_of_a_board_with_any_count():
    three_blocks = 1 << 2 | 1 << 8 | 1 << 13
    planes = grid_learners.encode_boards([0], [three_blocks])
    assert planes[0, 0].nonzero().tolist() == [[0, 0]]
    assert planes[0, 1].nonzero().tolist() == [[0, 2], [1, 2], [2, 1]]


def state_of(world, bot, block_cells):
    return world.index_by_key[0][sum(1 << cell for cell in block_cells) * 32 + bot]


def test_a_second_camera_ends_the_game_when_only_the_row_is_blocked():
    # On 5 x 4 the hole is cell 22. The robot on cell 10 pushes the block on cell 16 south into it, with a block on cell 19 on the bottom row.
    for cameras, game_goes_on in ((1, True), (2, False)):
        world = grid_learners.World([(5, 4, cameras)])
        state = state_of(world, 10, [16, 19])
        following = world.next_index[state, 1]
        assert world.rewards[state, 1] > 0
        assert (following >= 0) == game_goes_on


def test_the_robot_stops_at_the_edge_of_a_smaller_grid():
    world = grid_learners.World([(5, 4, 1)])
    east_edge, south_edge = state_of(world, 4, []), state_of(world, 18, [])
    assert world.next_index[east_edge, 3] == east_edge
    assert world.next_index[south_edge, 1] == south_edge


def test_a_mix_keeps_each_settings_own_rules_and_input():
    mix = grid_learners.World([(5, 4, 1), (6, 5, 2)])
    assert set(mix.setting_of[mix.tests]) == {0, 1}
    second_camera_plane = mix.encode(mix.tests).numpy()[:, 4]
    assert np.array_equal(second_camera_plane.any(axis=(1, 2)), mix.setting_of[mix.tests] == 1)
    second_setting_moves = mix.next_index[mix.setting_of == 1]
    assert (second_setting_moves[second_setting_moves >= 0] >= (mix.setting_of == 0).sum()).all()


def test_q_learning_on_the_smallest_grid_finds_the_exact_best_return():
    world = grid_learners.World([(5, 4, 1)])
    table = grid_learners.GridTable(seed=1, world=world)
    while table.steps < 40_000_000:
        table.iterate()
    learned, _, _ = grid_learners.evaluate_greedy(table.scores, world)
    assert learned > 0.8 * world.optimum["return"]
