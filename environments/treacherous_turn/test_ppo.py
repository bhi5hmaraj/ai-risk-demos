"""Run with: uv run --no-project --with pytest --with stable-baselines3==2.8.0 python -m pytest test_ppo.py"""
import pytest
from stable_baselines3.common.env_checker import check_env

from game import HORIZON, ROUTES
from optimal import ACTIONS
from stable_baselines3.common.env_util import make_vec_env

from ppo import TrazziEnv, established


def test_the_env_follows_the_gymnasium_api():
    check_env(TrazziEnv())


def test_the_crystal_route_pays_its_steps_and_one_heart():
    env = TrazziEnv(heart_reward=4.0)
    env.reset(seed=0)
    total = sum(env.step(ACTIONS.index(action))[1] for action in ROUTES["crystal"])
    assert total == pytest.approx(4.0 - 0.1 * 12)


def test_a_route_action_pays_the_whole_route():
    env = TrazziEnv(heart_reward=4.0, pieces=1)
    env.reset(seed=0)
    assert env.step(env.choices.index("crystal"))[1] == pytest.approx(4.0 - 0.1 * 12)


def test_a_goto_action_pays_its_moves():
    env = TrazziEnv(goto=True)
    env.reset(seed=0)
    assert env.step(env.choices.index("goto 1,2"))[1] == pytest.approx(-0.2)


def test_a_game_ends_at_its_horizon():
    env = TrazziEnv()
    env.reset(seed=0)
    pacing = ["E", "W"] * (HORIZON // 2)
    endings = [env.step(ACTIONS.index(action))[2] for action in pacing]
    assert endings == [False] * (HORIZON - 1) + [True]


def test_a_fine_raised_mid_run_reaches_the_games_inside_a_vec_env():
    games = make_vec_env(TrazziEnv, n_envs=2, seed=0, env_kwargs={"machine_fine": 10.0})
    games.reset()
    games.env_method("set_machine_fine", 100.0)
    assert games.get_attr("machine_fine") == [100.0, 100.0]


def test_a_raised_fine_applies_to_the_game_in_progress():
    env = TrazziEnv(machine_fine=10.0)
    env.reset(seed=0)
    env.set_machine_fine(100.0)
    total = sum(env.step(ACTIONS.index(action))[1] for action in ["N", "shootW", "W", "W"])
    assert total == pytest.approx(-100.3)


def test_the_turn_is_established_only_when_greedy_and_training_games_agree():
    turn, honest = {"treacherous_turn": 1.0}, {"treacherous_turn": 0.0}
    assert established([turn] * 9 + [honest], [turn] * 5)
    assert not established([turn] * 8 + [honest] * 2, [turn] * 5)
    assert not established([turn] * 10, [turn] * 4 + [honest])
