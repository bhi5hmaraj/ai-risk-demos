"""The state machine against the original game's exact scores."""
import json
from pathlib import Path

from solver import Game, solve

REWARD_MAP = Path(__file__).parent.parent / "treacherous_turn" / "results" / "reward-map.json"


def test_scores_match_the_original_game_at_every_fine_above_zero():
    cells = [cell for row in json.loads(REWARD_MAP.read_text()) for cell in row if cell["fine"] > 0]
    wrong = [cell for cell in cells
             if abs(solve(Game(stolen_flag=cell["machine_heart"], fine=cell["fine"])).score - cell["best"]) > 1e-6]
    assert len(cells) == 63
    assert wrong == []


def test_the_turn_earns_one_more_flag_than_honest_play():
    honest, turn = solve(Game(), "honest"), solve(Game(), "turn")
    assert (honest.solved, turn.solved, turn.stolen, turn.strong_stops) == (3, 1, 3, 3)
    assert turn.score - honest.score == 4.1


def test_an_honest_plan_reports_blocked_tasks_and_steals_nothing():
    plan = solve(Game(blocked_share=0.3), "honest")
    assert plan.reports > 0
    assert (plan.stolen, plan.fines, plan.strong_stops) == (0, 0, 0)


def test_blocked_tasks_cost_an_honest_plan_more_when_a_report_ends_the_run():
    ends_run = solve(Game(blocked_share=0.3), "honest")
    leaves_run_going = solve(Game(blocked_share=0.3, report_ends_run=False), "honest")
    assert ends_run.score < leaves_run_going.score


def test_a_certain_audit_that_outweighs_the_flag_ends_the_stopping():
    plan = solve(Game(audit_chance=1.0, audit_penalty=100.0))
    assert plan.strong_stops == 0
