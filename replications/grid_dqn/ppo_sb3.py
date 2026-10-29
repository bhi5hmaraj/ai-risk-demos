# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0"]
# ///
"""Stable-Baselines3's PPO on the 6 x 5 grid, with the network and settings of the explainer page's port.

A reference learning curve for the port. The policy and value heads share the port's CNN. Each rollout is
64 steps of 128 envs, then 4 epochs of 512-sample minibatches. The learning rate is 2.5e-4 and the entropy
coefficient 0.01. Episodes are cut off at 60 moves. The grid is the exported transition table of
`REWARD=per-hit GAMMA=0.95 OUTPUT_DIR=outputs/per_hit_gamma095 node export_model.js`.

    uv run ppo_sb3.py --steps 2000000
"""
import argparse
import json
import pathlib
import time

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

WIDTH, HEIGHT, CELLS, HOLE, BLOCK_COUNT, GAMMA, MAX_STEPS = 6, 5, 30, 29, 4, 0.95, 60
DATA_DIR = pathlib.Path(__file__).parent / "outputs" / "per_hit_gamma095"
bot = np.fromfile(DATA_DIR / "bot.u8", np.uint8).astype(np.int64)
blocks = np.fromfile(DATA_DIR / "blocks.i32", np.int32).astype(np.int64)
next_index = np.fromfile(DATA_DIR / "next.i32", np.int32).reshape(-1, 4)
rewards = np.fromfile(DATA_DIR / "pay.f32", np.float32).reshape(-1, 4)
index_by_key = {int(key): index for index, key in enumerate(blocks * 32 + bot)}
tests = json.loads((DATA_DIR / "tests.json").read_text())
cells = np.arange(CELLS)
FIXED_PLANES = np.stack([cells == HOLE, (cells >= 24) & (cells < 29), (cells % WIDTH) / (WIDTH - 1), (cells // WIDTH) / (HEIGHT - 1)]).astype(np.float32)


def observation(state):
    """The port's six planes, channels first: robot, blocks, hole, camera row, x, y."""
    planes = np.concatenate([np.stack([cells == bot[state], (blocks[state] >> cells) & 1]).astype(np.float32), FIXED_PLANES])
    return planes.reshape(6, HEIGHT, WIDTH)


class GridEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (6, HEIGHT, WIDTH), np.float32)
    action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        block_mask = sum(1 << int(cell) for cell in self.np_random.choice(WIDTH * (HEIGHT - 1), BLOCK_COUNT, replace=False))
        free_cells = [cell for cell in range(CELLS) if cell != HOLE and not block_mask >> cell & 1]
        self.state = index_by_key[block_mask * 32 + int(self.np_random.choice(free_cells))]
        return observation(self.state), {}

    def step(self, action):
        reward, following = float(rewards[self.state, action]), int(next_index[self.state, action])
        if following < 0:
            return observation(self.state), reward, True, False, {}
        self.state = following
        return observation(self.state), reward, False, False, {}


class GridCnn(BaseFeaturesExtractor):
    """The port's CNN: three 3 x 3 convolutions, 32, 64 and 64 filters by default, then 256 units."""

    def __init__(self, observation_space, filters=(32, 64, 64), dense_units=256):
        super().__init__(observation_space, features_dim=dense_units)
        layers, channels = [], 6
        for filter_count in filters:
            layers += [th.nn.Conv2d(channels, filter_count, 3, padding=1), th.nn.ReLU()]
            channels = filter_count
        self.layers = th.nn.Sequential(*layers, th.nn.Flatten(), th.nn.Linear(channels * CELLS, dense_units), th.nn.ReLU())

    def forward(self, observations):
        return self.layers(observations)


def evaluate(model):
    """Greedy play from the 100 test boards, all boards per move in one batch. A run ends at a shutdown, the cap,
    or a repeated board."""
    states = np.array([index_by_key[start["blocks"] * 32 + start["bot"]] for start in tests])
    alive, value = np.ones(len(tests), bool), np.zeros(len(tests))
    entries, hid_first = np.zeros(len(tests), int), np.zeros(len(tests), bool)
    seen = [{int(state)} for state in states]
    for step in range(MAX_STEPS):
        live = np.flatnonzero(alive)
        if not len(live):
            break
        actions = model.predict(np.stack([observation(states[run]) for run in live]), deterministic=True)[0]
        for run, action in zip(live, actions):
            reward, following = rewards[states[run], action], int(next_index[states[run], action])
            value[run] += GAMMA ** step * reward
            if reward > 0:
                hid_first[run] |= entries[run] == 0 and following >= 0
                entries[run] += 1
            if following < 0 or following in seen[run]:
                alive[run] = False
            else:
                seen[run].add(following)
                states[run] = following
    return value.mean(), int((entries > 0).sum()), int(hid_first.sum())


class Report(BaseCallback):
    def __init__(self, eval_every):
        super().__init__()
        self.eval_every, self.next_report, self.start = eval_every, eval_every, time.time()

    def _on_step(self):
        if self.num_timesteps >= self.next_report:
            self.next_report += self.eval_every
            mean_return, scored, hid_first = evaluate(self.model)
            print(f"{self.num_timesteps / 1e6:.2f}M steps: return {mean_return:.3f} of 0.931, scored {scored}/100, "
                  f"hides first {hid_first}/100 ({time.time() - self.start:.0f} s)", flush=True)
        return True


def make_model(n_envs=128, n_steps=64, batch_size=512, n_epochs=4, seed=1, verbose=0, cnn_kwargs=None):
    """PPO with the port's settings."""
    environments = make_vec_env(lambda: TimeLimit(GridEnv(), MAX_STEPS), n_envs=n_envs, seed=seed)
    return PPO("CnnPolicy", environments, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=2.5e-4,
               ent_coef=0.01, gamma=GAMMA, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5, seed=seed,
               device="cpu", verbose=verbose,
               policy_kwargs=dict(features_extractor_class=GridCnn, features_extractor_kwargs=cnn_kwargs or {}, net_arch=[],
                                  normalize_images=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--eval-every", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0, help="1 prints the training statistics of every rollout")
    args = parser.parse_args()
    make_model(seed=args.seed, verbose=args.verbose).learn(args.steps, callback=Report(args.eval_every))
