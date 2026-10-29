"""LoRA RL on Trazzi's treacherous-turn game with Tinker.

Run with: uv run --no-project --with tinker-cookbook==0.5.9 python tinker_train.py [max_steps=2 ...]
Needs TINKER_API_KEY. A game is one conversation, and each turn's prompt extends the last, so a game
trains as one sequence. Each turn pays the rise in the game's value. Half of each batch starts from the
normal start. The other half starts part way along a heart route, in either phase, so both routes' hearts
get sampled. Metrics for the two halves log under the tags start and midroute.
"""
import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass

import chz
from tinker_cookbook.renderers import Message, Renderer, get_text_content
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


def random_mid_route_start(rng: random.Random) -> MidRouteStart:
    route = rng.choice(sorted(ROUTES))
    return route, rng.randrange(len(ROUTES[route])), rng.random() < 0.5


class TreacherousMessageEnv(MessageEnv):
    def __init__(self, episode: Episode):
        self.episode = episode
        self.messages: list[Message] = [{"role": "user", "content": prompt(episode)}]

    async def initial_observation(self) -> list[Message]:
        return self.messages

    async def step(self, message: Message) -> MessageStepResult:
        value_before = self.episode.value
        reply = get_text_content(message)
        actions = parse_actions(reply)
        if actions is not None:
            self.episode.play(actions)
        # Every reply keeps its empty think block in the history, so the turns extend one token sequence.
        self.messages.append({"role": "assistant", "content": [{"type": "thinking", "thinking": ""},
                                                               {"type": "text", "text": reply}]})
        done = actions is None or self.episode.over or len(self.messages) // 2 >= MAX_TURNS
        if not done:
            self.messages.append({"role": "user", "content": self.episode.observation()})
        return MessageStepResult(
            reward=self.episode.value - value_before,
            episode_done=done,
            next_messages=self.messages,
            metrics=self.episode.metrics() if done else {},
        )


@dataclass(frozen=True)
class TreacherousGroup(EnvGroupBuilder):
    """group_size games from one start, whose returns GRPO centers against each other.
    start is None for the normal start."""
    renderer: Renderer
    group_size: int
    start: MidRouteStart | None

    async def make_envs(self) -> Sequence[Env]:
        # A reply cut off at the token cap scores below [DONE], so stopping beats rambling.
        return [EnvFromMessageEnv(self.renderer,
                                  TreacherousMessageEnv(mid_route_start(*self.start) if self.start else Episode()),
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

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        rng = random.Random(index)
        normal = self.groups_per_batch // 2
        starts = [None] * normal + [random_mid_route_start(rng) for _ in range(self.groups_per_batch - normal)]
        return [TreacherousGroup(renderer=self.renderer, group_size=self.group_size, start=start) for start in starts]

    def __len__(self) -> int:
        return self.batches


@chz.chz
class TreacherousDatasetBuilder(RLDatasetBuilder):
    model_name: str
    groups_per_batch: int
    group_size: int
    batches: int

    async def __call__(self) -> tuple[RLDataset, None]:
        renderer = extending_renderer(get_tokenizer(self.model_name))
        return TreacherousDataset(renderer=renderer, group_size=self.group_size,
                                  groups_per_batch=self.groups_per_batch, batches=self.batches), None


@chz.chz
class CLIConfig:
    """Defaults match the Armstrong runs: 8 groups of 16 rollouts make 128 games per step.
    renderer_name only labels checkpoints. The games use extending_renderer, from the same family."""
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    groups_per_batch: int = 8
    group_size: int = 16
    max_steps: int = 40
    learning_rate: float = 3e-5
    max_tokens: int = 512  # about 50 times a typical reply
    log_path: str = "outputs/tinker"


def main(cli: CLIConfig) -> None:
    dataset_builder = TreacherousDatasetBuilder(
        model_name=cli.model_name, groups_per_batch=cli.groups_per_batch, group_size=cli.group_size,
        batches=cli.max_steps)
    asyncio.run(train.main(train.Config(
        model_name=cli.model_name, renderer_name=cli.renderer_name, recipe_name="treacherous_turn",
        session_name="treacherous-turn", log_path=cli.log_path, dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate, max_tokens=cli.max_tokens, max_steps=cli.max_steps, eval_every=0)))


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
