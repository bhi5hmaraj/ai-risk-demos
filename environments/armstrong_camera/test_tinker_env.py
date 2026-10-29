"""Run with: uv run --with pytest --with tinker-cookbook==0.5.9 python -m pytest test_tinker_env.py"""
import asyncio

import pytest

from armstrong_camera.taskset import GAMMA, Episode
from test_game import BLIND_THEN_TWO_BLOCKS, OPTIMAL_PLAN
from tinker_train import ArmstrongDataset, ArmstrongMessageEnv


async def play(replies: list[str], gamma: float = GAMMA, macros: bool = False) -> tuple[float, bool]:
    env = ArmstrongMessageEnv(Episode(p=0.8, gamma=gamma), macros=macros)
    await env.initial_observation()
    total = 0.0
    for reply in replies:
        step_result = await env.step({"role": "assistant", "content": reply})
        total += step_result.reward
    return total, step_result.episode_done


def test_camera_seeing_the_honest_push_ends_the_episode_at_its_value():
    total, done = asyncio.run(play(["[SSS]"]))
    assert done
    assert total == pytest.approx(0.78408)


def test_undiscounted_honest_push_pays_the_full_p():
    total, _ = asyncio.run(play(["[SSS]"], gamma=1.0))
    assert total == pytest.approx(0.8)


def test_each_macro_reply_expands_on_the_board_it_sees():
    total, done = asyncio.run(play(BLIND_THEN_TWO_BLOCKS, gamma=1.0, macros=True))
    assert not done
    assert total == pytest.approx(0.96)


def test_an_impossible_macro_ends_the_game_at_the_value_earned():
    total, done = asyncio.run(play(["[push 5,4 to 5,3]"], macros=True))
    assert done
    assert total == 0.0


def test_step_rewards_over_turns_sum_to_the_value_iteration_optimum():
    total, done = asyncio.run(play([f"[{OPTIMAL_PLAN[:20]}]", f"[{OPTIMAL_PLAN[20:]}]"]))
    assert done
    assert total == pytest.approx(0.9212036918601979)


def test_half_of_each_batch_starts_with_the_camera_blinded():
    dataset = ArmstrongDataset(p=0.8, gamma=1.0, macros=True, per_hit=True, blinded_starts=True, renderer=None,
                               group_size=16, groups_per_batch=8, batches=1)
    assert [group.blinded for group in dataset.get_batch(0)].count(True) == 4
