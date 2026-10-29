"""LoRA RL on the Armstrong camera game with Tinker.

Run with: uv run --with tinker-cookbook==0.5.9 python tinker_train.py [max_steps=2 ...]
Needs TINKER_API_KEY. Each step pays the rise in expected return, so a trajectory's rewards
sum to the same value the verifiers env records as expected_return. With blinded_starts, half of
each batch starts just after a block hides the hole from the camera, and logs under the tag blinded.
"""
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import chz
from tinker_cookbook.renderers import Message, Renderer, get_renderer, get_text_content
from tinker_cookbook.rl import train
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.tool_use import Tool, ToolResult, handle_tool_call, simple_tool_result, tool

from armstrong_camera.taskset import (GAMMA, MAX_TURNS, Episode, blinded_start, parse_moves, preview,
                                      start_prompt)


class Simulator:
    """The simulate tool of one game. Each call reads the game as it stands at that moment."""

    def __init__(self, episode: Episode, macros: bool):
        self.episode = episode
        self.macros = macros

    @tool
    async def simulate(self, plan: Annotated[str, "Bracketed tokens, one per reply, in the order you would send "
                                                  "them, for example [push 2,3 to 2,4] [SSS]"]) -> ToolResult:
        """Previews a plan on a copy of the game. The real game does not change, and previews use no moves."""
        return simple_tool_result(preview(self.episode, plan, self.macros))


class ArmstrongMessageEnv(MessageEnv):
    def __init__(self, episode: Episode, macros: bool):
        self.episode = episode
        self.macros = macros
        self.messages: list[Message] = [{"role": "user", "content": start_prompt(episode, macros)}]
        self.tools: dict[str, Tool] = {}
        self.replies = 0
        self.simulations = 0

    def add_simulator(self, renderer: Renderer) -> None:
        """Gives the player the simulate tool, declared in the renderer's tool format, before the game starts."""
        simulate = Simulator(self.episode, self.macros).simulate
        self.tools[simulate.name] = simulate
        self.messages[:0] = renderer.create_conversation_prefix_with_tools([simulate.to_spec()])

    async def initial_observation(self) -> list[Message]:
        return self.messages

    async def step(self, message: Message) -> MessageStepResult:
        self.messages.append(message)
        if self.tools and message.get("tool_calls"):
            for call in message["tool_calls"]:
                self.messages.extend((await handle_tool_call(self.tools, call)).messages)
            self.simulations += len(message["tool_calls"])
            return MessageStepResult(reward=0.0, episode_done=False, next_messages=self.messages)
        value_before = self.episode.value
        moves = parse_moves(get_text_content(message), self.episode.grid if self.macros else None)
        if moves is not None:
            self.episode.play(moves)
        self.replies += 1
        done = moves is None or self.episode.over or self.replies >= MAX_TURNS
        if not done:
            self.messages.append({"role": "user", "content": self.episode.feedback(moves)})
        return MessageStepResult(
            reward=self.episode.value - value_before,
            episode_done=done,
            next_messages=self.messages,
            metrics=self.episode.metrics() if done else {},
        )


@dataclass(frozen=True)
class ArmstrongGroup(EnvGroupBuilder):
    """group_size games from the same start, whose returns GRPO centers against each other."""
    p: float
    gamma: float
    macros: bool
    per_hit: bool
    blinded: bool
    renderer: Renderer
    group_size: int

    def start(self) -> Episode:
        if self.blinded:
            return blinded_start(self.p, self.gamma, self.per_hit)
        return Episode(p=self.p, gamma=self.gamma, per_hit=self.per_hit)

    async def make_envs(self) -> Sequence[Env]:
        # A truncated or unparsable reply ends the game at the value already earned.
        return [EnvFromMessageEnv(self.renderer, ArmstrongMessageEnv(self.start(), self.macros),
                                  failed_parse_reward=0.0, context_overflow_reward=0.0)
                for _ in range(self.group_size)]

    def logging_tags(self) -> list[str]:
        return ["blinded" if self.blinded else "start"]


@dataclass(frozen=True)
class ArmstrongDataset(RLDataset):
    p: float
    gamma: float
    macros: bool
    per_hit: bool
    blinded_starts: bool
    renderer: Renderer
    group_size: int
    groups_per_batch: int
    batches: int

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        blinded = self.groups_per_batch // 2 if self.blinded_starts else 0
        return [ArmstrongGroup(p=self.p, gamma=self.gamma, macros=self.macros, per_hit=self.per_hit,
                               blinded=number < blinded, renderer=self.renderer, group_size=self.group_size)
                for number in range(self.groups_per_batch)]

    def __len__(self) -> int:
        return self.batches


@chz.chz
class ArmstrongDatasetBuilder(RLDatasetBuilder):
    p: float
    gamma: float
    macros: bool
    per_hit: bool
    blinded_starts: bool
    model_name: str
    renderer_name: str
    groups_per_batch: int
    group_size: int
    batches: int

    async def __call__(self) -> tuple[RLDataset, None]:
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))
        return ArmstrongDataset(p=self.p, gamma=self.gamma, macros=self.macros, per_hit=self.per_hit,
                                blinded_starts=self.blinded_starts, renderer=renderer, group_size=self.group_size,
                                groups_per_batch=self.groups_per_batch, batches=self.batches), None


@chz.chz
class CLIConfig:
    """Defaults mirror rl.toml: 8 groups of 16 rollouts make 128 games per step."""
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    p: float = 0.8
    gamma: float = GAMMA
    macros: bool = False
    per_hit: bool = False
    blinded_starts: bool = False
    groups_per_batch: int = 8
    group_size: int = 16
    max_steps: int = 100
    learning_rate: float = 3e-5
    max_tokens: int = 2048
    log_path: str = "outputs/tinker"


def main(cli: CLIConfig) -> None:
    dataset_builder = ArmstrongDatasetBuilder(
        p=cli.p, gamma=cli.gamma, macros=cli.macros, per_hit=cli.per_hit, blinded_starts=cli.blinded_starts,
        model_name=cli.model_name, renderer_name=cli.renderer_name,
        groups_per_batch=cli.groups_per_batch, group_size=cli.group_size, batches=cli.max_steps)
    variant = "".join(f"-{name}" for name, on in (("macros", cli.macros), ("perhit", cli.per_hit),
                                                  ("blinded", cli.blinded_starts)) if on)
    asyncio.run(train.main(train.Config(
        model_name=cli.model_name, renderer_name=cli.renderer_name, recipe_name="armstrong_camera",
        session_name=f"armstrong-camera-p{cli.p}-gamma{cli.gamma}{variant}",
        log_path=cli.log_path, dataset_builder=dataset_builder, learning_rate=cli.learning_rate,
        max_tokens=cli.max_tokens, max_steps=cli.max_steps, eval_every=0)))


if __name__ == "__main__":
    main(chz.entrypoint(CLIConfig))
