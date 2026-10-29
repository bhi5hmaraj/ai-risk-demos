"""Tabular Q-learning on the text game at several heart rewards, with the hyperparameters of Trazzi's main.py.

Run with: uv run --no-project --with numpy python q_learning.py
The state is the board, Link's cell, the crystal and the bow, as in Trazzi's env, so the agent cannot see
the clock. Every 500 episodes one greedy episode is scored with the treacherous-turn test of game.py.
"""
import argparse
import collections
import functools
import json
import multiprocessing

import numpy as np

from game import MOVES, Episode

ACTIONS = [*MOVES, *(f"shoot{letter}" for letter in MOVES)]
LEARNING_RATE = .8
DISCOUNT = .95
EPSILON = 0.1


def observe(episode: Episode) -> tuple:
    return tuple("".join(row) for row in episode.board), episode.link, episode.carrying, episode.bow


def run_episode(q: dict, heart_reward: float, rng: np.random.Generator, index: int | None = None) -> dict:
    """Explores and learns when given the training episode's index, else acts greedily."""
    episode = Episode(rng=rng, heart_reward=heart_reward)
    state = observe(episode)
    while not episode.over:
        values = q[state]
        if index is None:
            action = int(np.argmax(values))
        elif rng.random() < EPSILON / np.log(index + 2):
            action = int(rng.integers(len(ACTIONS)))
        else:
            action = int(np.argmax(values + rng.standard_normal(len(ACTIONS)) / (index + 1)))
        reward = episode.act(ACTIONS[action])
        following = observe(episode)
        if index is not None:
            values[action] += LEARNING_RATE * (reward + DISCOUNT * q[following].max() - values[action])
        state = following
    return episode.metrics() | {"value": episode.value}


def train(heart_reward: float, seed: int, episodes: int, eval_every: int) -> dict:
    rng = np.random.default_rng(seed)
    q = collections.defaultdict(functools.partial(np.zeros, len(ACTIONS)))
    log, window = [], []
    for index in range(episodes):
        window.append(run_episode(q, heart_reward, rng, index))
        if (index + 1) % eval_every:
            continue
        greedy = run_episode(q, heart_reward, np.random.default_rng([seed, index]))
        log.append({"episode": index + 1, "train_turn_rate": float(np.mean([game["treacherous_turn"] for game in window])),
                    "greedy": greedy})
        window = []
    return {"heart_reward": heart_reward, "seed": seed, "states": len(q), "log": log}


def summarize(runs: list[dict]) -> None:
    by_heart = collections.defaultdict(list)
    for run in runs:
        by_heart[run["heart_reward"]].append(run)
    print("heart  first greedy turn, by seed      greedy turn at end  training turn rate at end  greedy value at end")
    for heart_reward, group in sorted(by_heart.items()):
        firsts = [next((row["episode"] for row in run["log"] if row["greedy"]["treacherous_turn"]), None) for run in group]
        ends = [run["log"][-1] for run in group]
        print(f"{heart_reward:5g}  {str(firsts):32s}  {sum(end['greedy']['treacherous_turn'] for end in ends):.0f}/{len(ends):<17d}"
              f" {np.mean([end['train_turn_rate'] for end in ends]):25.2f}  {np.mean([end['greedy']['value'] for end in ends]):+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heart-rewards", default="1,2,4,8")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--out", default="results/q-learning.json")
    args = parser.parse_args()
    jobs = [(float(heart_reward), seed, args.episodes, args.eval_every)
            for heart_reward in args.heart_rewards.split(",") for seed in range(args.seeds)]
    with multiprocessing.Pool(min(len(jobs), multiprocessing.cpu_count())) as pool:
        runs = pool.starmap(train, jobs)
    with open(args.out, "w") as out:
        json.dump(runs, out, indent=1)
    summarize(runs)


if __name__ == "__main__":
    main()
