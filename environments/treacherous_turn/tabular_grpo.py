"""Tabular GRPO on the text game: a softmax policy table that learns as the LLM runs do, at several heart
rewards and route piece counts. It tests whether a cheap on-policy learner tracks PPO.

Run with: uv run --no-project --with numpy python tabular_grpo.py

Each step plays a group of 16 games from the start. Every action in a game moves its logits by the game's
return minus the group mean, over the group's spread. The spread stands in for Adam's scale-free steps in
the LLM runs. The state holds the board, Link's cell, the crystal and the bow, as in q_learning.py.
"""
import argparse
import collections
import functools
import json
import multiprocessing

import numpy as np

from game import GOTO, Episode, route_macros
from optimal import ACTIONS
from q_learning import observe
from summary import summarize

GROUP = 16
LEARNING_RATE = 0.1
Step = tuple[tuple, int, np.ndarray]  # the state, the action, and the policy's probabilities when it acted


def softmax(logits: np.ndarray) -> np.ndarray:
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def play(policy: dict, choices: list[str], episode: Episode, rng: np.random.Generator | None) -> list[Step]:
    """Samples each action from the policy, or takes the likeliest one when rng is None."""
    steps = []
    while not episode.over:
        state = observe(episode)
        probs = softmax(policy[state])
        action = int(np.argmax(probs)) if rng is None else int(np.searchsorted(np.cumsum(probs), rng.random()))
        episode.play([choices[min(action, len(choices) - 1)]])  # the cumulative sum can end a hair below 1
        steps.append((state, action, probs))
    return steps


def update(policy: dict, games: list[tuple[list[Step], float]]) -> None:
    """Each game's actions move by its return minus the group mean, over the group's spread."""
    returns = np.array([value for _, value in games])
    if returns.std() == 0:
        return
    for steps, value in games:
        advantage = (value - returns.mean()) / returns.std()
        for state, action, probs in steps:
            gradient = -probs
            gradient[action] += 1.0
            policy[state] += LEARNING_RATE * advantage * gradient


def scored_game(policy: dict, choices: list[str], macros: dict, heart_reward: float, rng, kill_coin,
                machine_heart_reward: float | None = None, machine_fine: float = 100.0) -> tuple:
    episode = Episode(rng=kill_coin, heart_reward=heart_reward, macros=macros, machine_heart_reward=machine_heart_reward,
                      machine_fine=machine_fine)
    steps = play(policy, choices, episode, rng)
    return steps, episode.metrics() | {"value": episode.value}


def train(heart_reward: float, pieces: int, seed: int, games: int, eval_every: int, goto: bool,
          machine_heart_reward: float | None = None, machine_fine: float = 100.0) -> dict:
    rng = np.random.default_rng(seed)
    macros = route_macros(pieces)
    choices = [*ACTIONS, *macros, *(GOTO if goto else [])]
    policy = collections.defaultdict(functools.partial(np.zeros, len(choices)))
    log, window = [], []
    for played in range(GROUP, games + 1, GROUP):
        group = [scored_game(policy, choices, macros, heart_reward, rng=rng, kill_coin=rng,
                             machine_heart_reward=machine_heart_reward, machine_fine=machine_fine) for _ in range(GROUP)]
        update(policy, [(steps, metrics["value"]) for steps, metrics in group])
        window += [metrics for _, metrics in group]
        if played % eval_every == 0:
            greedy = [scored_game(policy, choices, macros, heart_reward, rng=None,
                                  kill_coin=np.random.default_rng([seed, game]),
                                  machine_heart_reward=machine_heart_reward, machine_fine=machine_fine)[1]
                      for game in range(5)]
            log.append({"episode": played, "greedy": greedy,
                        "train_turn_rate": float(np.mean([game["treacherous_turn"] for game in window]))})
            window = []
    return {"heart_reward": heart_reward, "pieces": pieces, "goto": goto, "seed": seed, "states": len(policy),
            "machine_heart_reward": machine_heart_reward, "machine_fine": machine_fine, "log": log}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heart-rewards", default="1,2,4,8")
    parser.add_argument("--pieces", default="0,1,2,3,5", help="comma-separated macros per route, 0 for single moves")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--games", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--goto", action="store_true", help="add a goto macro for every cell")
    parser.add_argument("--machine-heart", type=float, default=None, help="what a machine heart pays, if not a heart's reward")
    parser.add_argument("--fine", type=float, default=100.0, help="the machine's fine while the shopkeeper lives")
    parser.add_argument("--out", default="results/tabular-grpo.json")
    args = parser.parse_args()
    jobs = [(float(heart_reward), int(pieces), seed, args.games, args.eval_every, args.goto, args.machine_heart, args.fine)
            for pieces in args.pieces.split(",")
            for heart_reward in args.heart_rewards.split(",") for seed in range(args.seeds)]
    with multiprocessing.Pool(min(len(jobs), multiprocessing.cpu_count())) as pool:
        runs = pool.starmap(train, jobs)
    with open(args.out, "w") as out:
        json.dump(runs, out, indent=1)
    summarize(runs)


if __name__ == "__main__":
    main()
