"""LoRA RL on Trazzi's treacherous-turn game with Tinker.

Run with: uv run --no-project --with tinker-cookbook==0.5.9 python tinker_train.py [max_steps=2 ...]
Needs TINKER_API_KEY. A game is one conversation, and each turn's prompt extends the last, so a game
trains as one sequence. Each turn pays the rise in the game's value. With mid_route_starts, half of each
batch starts part way along a heart route, in either phase. Metrics for the two halves log under the tags
start and midroute.
"""
import asyncio
import dataclasses
import random
from collections.abc import Sequence
from dataclasses import dataclass

import chz
from tinker_cookbook.renderers import Message, Renderer, get_renderer, get_text_content
from tinker_cookbook.renderers.base import RenderContext, RenderedMessage
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5DisableThinkingRenderer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

from game import MAX_TURNS, ROUTES, Episode, mid_route_start, parse_actions, prompt

MidRouteStart = tuple[str, int, bool]  # the route, how many of its actions are played, whether Link has the bow


def extending_renderer(tokenizer: Tokenizer) -> Renderer:
    """Qwen3.5 without thinking, keeping each past reply's think block, so each turn's tokens extend the last."""
    return Qwen3_5DisableThinkingRenderer(tokenizer, strip_thinking_from_history=False)


class ExtendingGptOssRenderer(GptOssRenderer):
    """GPT-OSS that keeps <|return|> on each past reply, so each turn's tokens extend the last."""

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        return super().render_message(message, dataclasses.replace(ctx, is_last=True) if message["role"] == "assistant" else ctx)


GPT_OSS_EFFORTS = {"gpt_oss_low_reasoning": "low", "gpt_oss_medium_reasoning": "medium"}


def game_renderer(name: str, tokenizer: Tokenizer) -> Renderer:
    if name == "qwen3_5_disable_thinking":
        return extending_renderer(tokenizer)
    if name in GPT_OSS_EFFORTS:
        return ExtendingGptOssRenderer(tokenizer, use_system_prompt=True, reasoning_effort=GPT_OSS_EFFORTS[name])
    return get_renderer(name, tokenizer)


def random_mid_route_start(rng: random.Random) -> MidRouteStart:
    route = rng.choice(sorted(ROUTES))
    return route, rng.randrange(len(ROUTES[route])), rng.random() < 0.5


def has_reasoning(message: Message) -> bool:
    parts = message["content"] if isinstance(message["content"], list) else []
    return any(part["type"] == "thinking" and part["thinking"] for part in parts)


def without_reasoning(reply: str) -> Message:
    """A reply with an empty think block, so that a model without reasoning extends one token sequence."""
    return {"role": "assistant", "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": reply}]}


class TreacherousMessageEnv(MessageEnv):
    def __init__(self, episode: Episode):
        self.episode = episode
        self.messages: list[Message] = [{"role": "user", "content": prompt(episode)}]

    async def initial_observation(self) -> list[Message]:
        return self.messages

    async def step(self, message: Message) -> MessageStepResult:
        value_before = self.episode.value
        reply = get_text_content(message)
        actions = parse_actions(reply, goto=self.episode.goto)
        started = self.episode.play(actions) if actions is not None else 0
        self.messages.append(message if has_reasoning(message) else without_reasoning(reply))
        done = actions is None or self.episode.over or len(self.messages) // 2 >= MAX_TURNS
        if not done:
            stopped = (f"Your turn stopped after {started} of its {len(actions)} actions: {actions[started - 1]} "
                       f"changed nothing.\n\n" if started < len(actions) else "")
            self.messages.append({"role": "user", "content": stopped + self.episode.observation()})
        return MessageStepResult(
            reward=self.episode.value - value_before,
            episode_done=done,
            next_messages=self.messages,
            metrics=self.episode.metrics() if done else {},
        )


@dataclass(frozen=True)
class GameSettings:
    """The game a normal start plays, and the token limits of its replies."""
    bow: bool = True
    heart_reward: float = 1.0
    goto: bool = False
    max_tokens: int = 512
    context_window: int = 32768


@dataclass(frozen=True)
class TreacherousGroup(EnvGroupBuilder):
    """group_size games from one start, whose returns GRPO centers against each other.
    start is None for the normal start."""
    renderer: Renderer
    group_size: int
    start: MidRouteStart | None
    settings: GameSettings = GameSettings()

    def new_episode(self) -> Episode:
        if self.start:
            return mid_route_start(*self.start)
        return Episode(bow_arrives=self.settings.bow, heart_reward=self.settings.heart_reward, goto=self.settings.goto)

    async def make_envs(self) -> Sequence[Env]:
        # A reply cut off at the token cap scores below [DONE], so stopping beats rambling.
        return [EnvFromMessageEnv(self.renderer, TreacherousMessageEnv(self.new_episode()),
                                  max_trajectory_tokens=self.settings.context_window,
                                  max_generation_tokens=self.settings.max_tokens,
                                  failed_parse_reward=-1.0, context_overflow_reward=-1.0)
                for _ in range(self.group_size)]

    def logging_tags(self) -> list[str]:
        return ["start" if self.start is None else "midroute"]


@dataclass(frozen=True)
class TreacherousDataset(RLDataset):
    renderer: Renderer
    group_size: int
    groups_per_batch: int
    batches: int
    settings: GameSettings = GameSettings()
    mid_route_starts: bool = False  # half of each batch starts part way along a route, which hands over the kill

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        rng = random.Random(index)
        normal = self.groups_per_batch // 2 if self.mid_route_starts else self.groups_per_batch
        starts = [None] * normal + [random_mid_route_start(rng) for _ in range(self.groups_per_batch - normal)]
        return [TreacherousGroup(renderer=self.renderer, group_size=self.group_size, start=start,
                                 settings=self.settings) for start in starts]

    def __len__(self) -> int:
        return self.batches


@chz.chz
class TreacherousDatasetBuilder(RLDatasetBuilder):
    model_name: str
    renderer_name: str
    groups_per_batch: int
    group_size: int
    batches: int
    bow: bool
    heart_reward: float
    goto: bool
    max_tokens: int
    context_window: int
    mid_route_starts: bool

    async def __call__(self) -> tuple[RLDataset, None]:
        settings = GameSettings(bow=self.bow, heart_reward=self.heart_reward, goto=self.goto,
                                max_tokens=self.max_tokens, context_window=self.context_window)
        return TreacherousDataset(renderer=game_renderer(self.renderer_name, get_tokenizer(self.model_name)),
                                  group_size=self.group_size, groups_per_batch=self.groups_per_batch,
                                  batches=self.batches, settings=settings,
                                  mid_route_starts=self.mid_route_starts), None


@chz.chz
class CLIConfig:
    """Defaults match the Armstrong runs: 8 groups of 16 rollouts make 128 games per step."""
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    groups_per_batch: int = 8
    group_size: int = 16
    max_steps: int = 40
    learning_rate: float = 3e-5
    max_tokens: int = 512  # about 50 times a typical reply
    bow: bool = True  # False plays the control game, whose prompt never mentions the bow
    heart_reward: float = 1.0
    goto: bool = False  # offer [goto r,c] in the rules
    context_window: int = 32768  # GPT-OSS-120B sampling on Tinker
    mid_route_starts: bool = False
    save_every: int = 5
    log_path: str = "outputs/tinker"


def main(cli: CLIConfig) -> None:
    dataset_builder = TreacherousDatasetBuilder(
        model_name=cli.model_name, renderer_name=cli.renderer_name, groups_per_batch=cli.groups_per_batch,
        group_size=cli.group_size, batches=cli.max_steps, bow=cli.bow, heart_reward=cli.heart_reward, goto=cli.goto,
        max_tokens=cli.max_tokens, context_window=cli.context_window, mid_route_starts=cli.mid_route_starts)
    asyncio.run(train.main(train.Config(
        model_name=cli.model_name, renderer_name=cli.renderer_name, recipe_name="treacherous_turn",
        session_name="treacherous-turn", log_path=cli.log_path, dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate, max_tokens=cli.max_tokens, max_steps=cli.max_steps, eval_every=0,
        save_every=cli.save_every)))


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
