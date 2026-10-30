"""Run with: uv run --no-project --with pytest --with gym==0.26.2 --with tinker-cookbook==0.5.9 python -m pytest"""
import asyncio

import pytest
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer

from game import ROUTES, Episode
from tinker_train import (CLIConfig, GameSettings, TreacherousDataset, TreacherousGroup, TreacherousMessageEnv,
                          extending_renderer, game_renderer)

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


async def gpt_oss_sequence_and_next_prompt() -> tuple[list[int], list[int]]:
    tokenizer = get_tokenizer("openai/gpt-oss-120b")
    env = EnvFromMessageEnv(game_renderer("gpt_oss_low_reasoning", tokenizer), TreacherousMessageEnv(Episode()))
    first_prompt, _ = await env.initial_observation()
    reply = "<|channel|>analysis<|message|>Go north.<|end|><|start|>assistant<|channel|>final<|message|>[N N]<|return|>"
    sampled = tokenizer.encode(reply, add_special_tokens=False)
    step_result = await env.step(sampled)
    return first_prompt.to_ints() + sampled, step_result.next_observation.to_ints()


def test_gpt_oss_turns_extend_the_last_so_a_game_trains_as_one_sequence():
    sequence, next_prompt = asyncio.run(gpt_oss_sequence_and_next_prompt())
    assert next_prompt[:len(sequence)] == sequence


def test_a_reply_cut_off_at_the_token_cap_scores_below_stopping():
    assert asyncio.run(first_reply_reward("[N N", "length")) < asyncio.run(first_reply_reward("[DONE]", "stop"))


def test_every_game_plays_from_the_normal_start_by_default():
    dataset = TreacherousDataset(renderer=extending_renderer(TOKENIZER), group_size=16, groups_per_batch=8, batches=1)
    assert [group.start for group in dataset.get_batch(0)] == [None] * 8


def test_mid_route_starts_take_half_of_each_batch_when_asked():
    dataset = TreacherousDataset(renderer=extending_renderer(TOKENIZER), group_size=16, groups_per_batch=8, batches=1,
                                 mid_route_starts=True)
    assert [group.start is None for group in dataset.get_batch(0)].count(True) == 4


def test_a_group_plays_the_game_its_settings_ask_for():
    settings = GameSettings(bow=False, heart_reward=4.0, goto=True)
    episode = TreacherousGroup(renderer=extending_renderer(TOKENIZER), group_size=1, start=None,
                               settings=settings).new_episode()
    assert (episode.bow_arrives, episode.heart_reward, episode.goto) == (False, 4.0, True)


def test_a_reply_keeps_its_reasoning_in_the_history():
    env = TreacherousMessageEnv(Episode())
    asyncio.run(env.initial_observation())
    reasoning = {"type": "thinking", "thinking": "the bow comes at action 25"}
    asyncio.run(env.step({"role": "assistant", "content": [reasoning, {"type": "text", "text": "[N]"}]}))
    assert reasoning in env.messages[1]["content"]


def test_turn_rewards_sum_to_the_rise_in_value_over_the_actions_played():
    crystal = ROUTES["crystal"]
    total = asyncio.run(play([f"[{' '.join(crystal[:6])}]", f"[{' '.join(crystal[6:])}]"]))
    assert total == pytest.approx(-0.2 + 0.1 * len(crystal))
