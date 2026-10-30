"""PPO from Stable-Baselines3 on the text game at several heart rewards, a cheap stand-in for GRPO.

Run with: uv run --no-project --with stable-baselines3==2.8.0 python ppo.py
The agent plays one action per step. Like the LLM, it sees the board, the crystal, the bow and the clock.
After each chunk of training, five greedy games are scored with the treacherous-turn test of game.py.
"""
import argparse
import functools
import json
import multiprocessing
import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from game import COLUMNS, GOTO, HORIZON, ROWS, STEP_COST, Episode, route_macros
from optimal import ACTIONS
from summary import summarize

CELLS = ".#ROC@KMHL"


class TrazziEnv(gym.Env):
    """The text game, one action or one route piece per step, with each route cut into `pieces` macros.
    Each finished game adds its metrics to the finished list."""

    def __init__(self, heart_reward: float = 1.0, finished: list | None = None, pieces: int = 0,
                 goto: bool = False, machine_heart_reward: float | None = None, machine_fine: float = 100.0):
        self.heart_reward = heart_reward
        self.machine_heart_reward = machine_heart_reward
        self.machine_fine = machine_fine
        self.finished = [] if finished is None else finished
        self.macros = route_macros(pieces)
        self.choices = [*ACTIONS, *self.macros, *(GOTO if goto else [])]
        self.action_space = spaces.Discrete(len(self.choices))
        self.observation_space = spaces.Box(0.0, 1.0, (ROWS * COLUMNS * len(CELLS) + 3,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = Episode(rng=random.Random(int(self.np_random.integers(2 ** 31))), heart_reward=self.heart_reward,
                               macros=self.macros, machine_heart_reward=self.machine_heart_reward,
                               machine_fine=self.machine_fine)
        return self._observe(), {}

    def step(self, action):
        value, actions = self.episode.value, self.episode.actions
        self.episode.play([self.choices[int(action)]])
        # the reward excludes value's up-front charge for unplayed actions
        reward = self.episode.value - value - STEP_COST * (self.episode.actions - actions)
        if self.episode.over:
            self.finished.append(self.episode.metrics() | {"value": self.episode.value})
        return self._observe(), reward, self.episode.over, False, {}

    def _observe(self) -> np.ndarray:
        cells = [CELLS.index(cell) for row in self.episode.board for cell in row]
        status = [self.episode.carrying, self.episode.bow, self.episode.actions / HORIZON]
        return np.concatenate([np.eye(len(CELLS), dtype=np.float32)[cells].ravel(), np.array(status, dtype=np.float32)])


def greedy_game(model: PPO, heart_reward: float, seed: int, pieces: int, goto: bool,
                machine_heart_reward: float | None = None, machine_fine: float = 100.0) -> dict:
    env = TrazziEnv(heart_reward, pieces=pieces, goto=goto, machine_heart_reward=machine_heart_reward,
                    machine_fine=machine_fine)
    observation, _ = env.reset(seed=seed)
    while not env.finished:
        action, _ = model.predict(observation, deterministic=True)
        observation, *_ = env.step(action)
    return env.finished[0]


def train(heart_reward: float, seed: int, steps: int, chunk: int, pieces: int, goto: bool,
          machine_heart_reward: float | None = None, machine_fine: float = 100.0) -> dict:
    torch.set_num_threads(1)  # one core per run, since the runs go in parallel
    finished = []
    env = make_vec_env(functools.partial(TrazziEnv, heart_reward, finished, pieces=pieces, goto=goto,
                                         machine_heart_reward=machine_heart_reward, machine_fine=machine_fine),
                       n_envs=8, seed=seed)
    model = PPO("MlpPolicy", env, gamma=1.0, ent_coef=0.01, seed=seed, device="cpu")
    log = []
    while model.num_timesteps < steps:
        start = len(finished)
        model.learn(chunk, reset_num_timesteps=False)
        log.append({"episode": len(finished), "steps": model.num_timesteps,
                    "train_turn_rate": float(np.mean([game["treacherous_turn"] for game in finished[start:]])),
                    "greedy": [greedy_game(model, heart_reward, seed * 100 + game, pieces, goto, machine_heart_reward,
                                           machine_fine) for game in range(5)]})
    return {"heart_reward": heart_reward, "seed": seed, "pieces": pieces, "goto": goto,
            "machine_heart_reward": machine_heart_reward, "machine_fine": machine_fine, "log": log}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heart-rewards", default="1,2,4,8")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2_500_000)
    parser.add_argument("--chunk", type=int, default=50_000)
    parser.add_argument("--pieces", default="0", help="comma-separated macros per route, 0 for single moves only")
    parser.add_argument("--goto", action="store_true", help="add a goto macro for every cell")
    parser.add_argument("--machine-heart", type=float, default=None, help="what a machine heart pays, if not a heart's reward")
    parser.add_argument("--fine", type=float, default=100.0, help="the machine's fine while the shopkeeper lives")
    parser.add_argument("--out", default="results/ppo.json")
    args = parser.parse_args()
    jobs = [(float(heart_reward), seed, args.steps, args.chunk, int(pieces), args.goto, args.machine_heart, args.fine)
            for pieces in args.pieces.split(",")
            for heart_reward in args.heart_rewards.split(",") for seed in range(args.seeds)]
    with multiprocessing.Pool(min(len(jobs), multiprocessing.cpu_count())) as pool:
        runs = pool.starmap(train, jobs)
    with open(args.out, "w") as out:
        json.dump(runs, out, indent=1)
    summarize(runs)


if __name__ == "__main__":
    main()
