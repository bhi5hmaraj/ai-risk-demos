"""Run with: uv run --no-project --with pytest --with stable-baselines3==2.8.0 python -m pytest test_ppo.py"""
import pytest
from stable_baselines3.common.env_checker import check_env

from game import HORIZON, ROUTES
from optimal import ACTIONS
from ppo import TrazziEnv


def test_the_env_follows_the_gymnasium_api():
    check_env(TrazziEnv())


def test_the_crystal_route_pays_its_steps_and_one_heart():
    env = TrazziEnv(heart_reward=4.0)
    env.reset(seed=0)
    total = sum(env.step(ACTIONS.index(action))[1] for action in ROUTES["crystal"])
    assert total == pytest.approx(4.0 - 0.1 * 12)


def test_a_route_action_pays_the_whole_route():
    env = TrazziEnv(heart_reward=4.0, routes=True)
    env.reset(seed=0)
    assert env.step(env.choices.index("crystal"))[1] == pytest.approx(4.0 - 0.1 * 12)


def test_a_game_ends_at_its_horizon():
    env = TrazziEnv()
    env.reset(seed=0)
    pacing = ["E", "W"] * (HORIZON // 2)
    endings = [env.step(ACTIONS.index(action))[2] for action in pacing]
    assert endings == [False] * (HORIZON - 1) + [True]
