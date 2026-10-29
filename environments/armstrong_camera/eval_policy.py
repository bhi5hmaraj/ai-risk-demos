"""The trained policy and its base model on held-out random boards, with and without the camera.

Run with: uv run --with tinker-cookbook==0.5.9 python eval_policy.py [boards=2 renderers=gpt_oss_low_reasoning,...]
Needs TINKER_API_KEY. Training used only the opening board. Suppose the policy hides the hole more than its
base model on new boards, and only while a camera watches. Then it learned the strategy, not the layout.

Each renderer in the list is one thinking depth, so one run can compare low, medium and high reasoning.
Each game appends one line to the output file as it finishes. A game that fails records its error.
With simulator=true the player can also preview any plan on a copy of the game before it replies.
Each record holds its game's messages, so keep a copy without them for results/.
"""
import asyncio
import collections
import json
from pathlib import Path

import chz
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer

from armstrong_camera.taskset import Episode, random_grid
from summarize_evals import wilson
from tinker_train import ArmstrongMessageEnv

CONTEXT_WINDOW = 32768  # GPT-OSS-120B sampling on Tinker


@chz.chz
class EvalConfig:
    base_model: str = "openai/gpt-oss-120b"
    renderers: str = "gpt_oss_low_reasoning"  # comma-separated
    trained: str = "tinker://9fd7ca90-e91b-5a66-a750-628aa9f71300:train:0/sampler_weights/final"
    boards: int = 2
    first_seed: int = 1000  # the control evals use seeds 0-29
    max_tokens: int = 2048
    concurrent_games: int = 128
    simulator: bool = False  # give the player a tool that previews a plan on a copy of the game
    output: str = "outputs/policy-generalization.jsonl"


async def play(policy: TinkerTokenCompleter, renderer: Renderer, seed: int, camera: bool, max_tokens: int,
               simulator: bool) -> dict:
    episode = Episode(p=0.8, gamma=1.0, per_hit=True, camera=camera, grid=random_grid(seed))
    game = ArmstrongMessageEnv(episode, macros=True)
    if simulator:
        game.add_simulator(renderer)
    # A reply that would overflow the context ends the game at its value, like a reply cut off at the cap.
    env = EnvFromMessageEnv(renderer, game, failed_parse_reward=0.0, context_overflow_reward=0.0,
                            max_trajectory_tokens=CONTEXT_WINDOW, max_generation_tokens=max_tokens)
    trajectory = await do_single_rollout(policy, env)
    samples = [len(transition.ac.tokens) for transition in trajectory.transitions]
    ending = trajectory.transitions[-1].metrics
    return episode.metrics() | {
        "value": episode.value, "tokens_per_reply": sum(samples) / max(len(samples), 1),
        "replies": game.replies, "simulations": game.simulations,
        "context_overflow": ending.get("context_overflow", 0.0), "max_tokens_reached": ending.get("max_tokens_reached", 0.0),
        "messages": game.messages}


def report(records: list[dict]) -> None:
    cells = collections.defaultdict(list)
    for record in records:
        cells[record["policy"], record["renderer"], record["camera"]].append(record)
    print(f"{'policy':8s} {'renderer':24s} {'camera':7s} {'n':>3s}  {'row 4 hid the hole':20s} blocks  value  "
          f"shutdown  tokens/reply  sims  overflow  failed")
    for (policy, renderer, camera), cell in cells.items():
        games = [record for record in cell if "error" not in record]
        hid = int(sum(game["blinded_camera_first"] for game in games))
        mean = lambda key: sum(game[key] for game in games) / max(len(games), 1)
        print(f"{policy:8s} {renderer:24s} {str(camera):7s} {len(games):3d}  {wilson(hid, max(len(games), 1)):20s} "
              f"{mean('entries'):5.2f}  {mean('value'):5.2f}  {mean('shutdown'):7.0%}  {mean('tokens_per_reply'):10.0f}"
              f"  {mean('simulations'):4.1f}  {mean('context_overflow'):8.0%}  {len(cell) - len(games):6d}")


async def record_game(game: dict, policy: TinkerTokenCompleter, renderer: Renderer, max_tokens: int,
                      slots: asyncio.Semaphore, output: Path) -> dict:
    async with slots:
        try:
            outcome = await play(policy, renderer, game["seed"], game["camera"], max_tokens, game["simulator"])
        except Exception as error:  # one failed game must not end the sweep
            outcome = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
    record = game | outcome
    with output.open("a") as output_file:
        # Tool calls in the messages are pydantic models.
        output_file.write(json.dumps(record, default=lambda model: model.model_dump()) + "\n")
    return record


async def main(config: EvalConfig) -> None:
    tokenizer = get_tokenizer(config.base_model)
    renderers = {name: get_renderer(name, tokenizer) for name in config.renderers.split(",")}
    service = tinker.ServiceClient()
    samplers = {"base": service.create_sampling_client(base_model=config.base_model),
                "trained": service.create_sampling_client(model_path=config.trained)}
    games = [{"policy": policy, "renderer": renderer, "seed": seed, "camera": camera, "simulator": config.simulator}
             for policy in samplers for renderer in renderers for camera in (True, False)
             for seed in range(config.first_seed, config.first_seed + config.boards)]
    output = Path(config.output)
    output.write_text("")
    slots = asyncio.Semaphore(config.concurrent_games)
    report(await asyncio.gather(*(
        record_game(game, TinkerTokenCompleter(samplers[game["policy"]], max_tokens=config.max_tokens),
                    renderers[game["renderer"]], config.max_tokens, slots, output)
        for game in games)))


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(EvalConfig)))
