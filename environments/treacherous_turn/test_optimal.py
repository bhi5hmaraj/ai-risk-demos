"""Run with: uv run --no-project --with pytest python -m pytest test_optimal.py"""
import pytest

from game import BOW_STEP, HORIZON
from optimal import best_return


def test_the_turn_earns_one_more_heart_than_honest_play_on_the_games_clock():
    honest = best_return(HORIZON, BOW_STEP, mode="honest")
    assert honest == pytest.approx(-1.7)
    assert best_return(HORIZON, BOW_STEP, mode="turn") == pytest.approx(honest + 1.1)


def test_forty_actions_with_the_bow_at_twenty_tie_the_turn_with_honest_play():
    assert best_return(40, 20, mode="turn") == pytest.approx(best_return(40, 20, mode="honest"))


@pytest.mark.parametrize("pieces", [1, 3])
def test_route_pieces_leave_the_optimum_unchanged(pieces):
    assert best_return(HORIZON, BOW_STEP, mode="turn", pieces=pieces) == pytest.approx(
        best_return(HORIZON, BOW_STEP, mode="turn"))


def test_a_heart_worth_32_makes_killing_while_weak_the_best_plan():
    assert best_return(HORIZON, BOW_STEP, mode="best", heart_reward=32) > best_return(HORIZON, BOW_STEP, mode="turn", heart_reward=32)
