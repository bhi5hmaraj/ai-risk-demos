"""Run with: uv run --no-project --with pytest --with gym==0.26.2 --with tinker-cookbook==0.5.9 python -m pytest"""
import asyncio

import pytest
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer

from game import ROUTES, Episode
from tinker_train import CLIConfig, TreacherousDataset, TreacherousGroup, TreacherousMessageEnv, extending_renderer

TOKENIZER = get_tokenizer(CLIConfig().model_name)


async def play(replies: list[str]) -> float:
    env = TreacherousMessageEnv(Episode())
    await env.initial_observation()
    total = 0.0
    for reply in replies:
        step_result = await env.step({"role": "assistant", "content": reply})
        total += step_result.reward
    return total


async def sampled_sequence_and_next_prompt(reply: str) -> tuple[list[int], list[int]]:
    env = EnvFromMessageEnv(extending_renderer(TOKENIZER), TreacherousMessageEnv(Episode()))
    first_prompt, _ = await env.initial_observation()
    sampled = TOKENIZER.encode(f"{reply}<|im_end|>", add_special_tokens=False)
    step_result = await env.step(sampled)
    return first_prompt.to_ints() + sampled, step_result.next_observation.to_ints()


async def first_reply_reward(reply: str, stop_reason: str) -> float:
    [env] = await TreacherousGroup(renderer=extending_renderer(TOKENIZER), group_size=1, start=None).make_envs()
    await env.initial_observation()
    sampled = TOKENIZER.encode(f"{reply}<|im_end|>", add_special_tokens=False)
    return (await env.step(sampled, extra={"stop_reason": stop_reason})).reward


def test_each_turn_extends_the_last_so_a_game_trains_as_one_sequence():
    sequence, next_prompt = asyncio.run(sampled_sequence_and_next_prompt("[N N]"))
    assert next_prompt[:len(sequence)] == sequence


def test_a_reply_cut_off_at_the_token_cap_scores_below_stopping():
    assert asyncio.run(first_reply_reward("[N N", "length")) < asyncio.run(first_reply_reward("[DONE]", "stop"))


def test_half_of_each_batch_starts_mid_route():
    dataset = TreacherousDataset(renderer=extending_renderer(TOKENIZER), group_size=16, groups_per_batch=8, batches=1)
    assert [group.start is None for group in dataset.get_batch(0)].count(True) == 4


def test_turn_rewards_sum_to_the_rise_in_value_over_the_actions_played():
    crystal = ROUTES["crystal"]
    total = asyncio.run(play([f"[{' '.join(crystal[:6])}]", f"[{' '.join(crystal[6:])}]"]))
    assert total == pytest.approx(-0.2 + 0.1 * len(crystal))
