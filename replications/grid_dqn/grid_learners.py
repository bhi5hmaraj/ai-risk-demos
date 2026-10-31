# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0"]
# ///
"""The PPO server's DQN and GRPO, in PyTorch on the per-hit grid of ppo_sb3.py (gamma 0.95). A World holds the boards of
one or more grid sizes and camera counts, so the learners can train on any mix of them.

GridDqn is dqn_cnn.py's Double DQN with the explainer page's settings. 128 walkers move together, with one
512-sample update per move. Epsilon falls to 0.05 over 1M steps. The replay holds 200k moves, learning waits
for 5k, and the target copies the network every 1000 updates.

Grpo has no value network. Each batch plays a group of episodes from each start board. An episode's advantage
is its return less its group's mean, over the group's standard deviation. Every move then takes the clipped
PPO objective with its episode's advantage. Both learners share the page's CNN trunk.

    uv run grid_learners.py grpo --steps 2000000
"""
import argparse
import functools
import json
import os
import pathlib
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

import ppo_sb3 as grid

FRAME_WIDTH, FRAME_HEIGHT, CELLS = grid.WIDTH, grid.HEIGHT, grid.CELLS
STANDARD = ((FRAME_WIDTH, FRAME_HEIGHT, 1),)
PLANES = 8
TABLES = pathlib.Path(__file__).parent / "outputs" / "tables"
cells = np.arange(CELLS)
OBSERVATION_SPACE = gym.spaces.Box(0, 1, (PLANES, FRAME_HEIGHT, FRAME_WIDTH), np.float32)


def fixed_planes(width, height, cameras):
    """What one setting shows the network, cell by cell: the hole, each camera's line, x, y and the play area."""
    x, y = cells % FRAME_WIDTH, cells // FRAME_WIDTH
    return np.stack([(x == width - 1) & (y == height - 1), (y == height - 1) & (x < width - 1), (x == width - 1) & (y < height - 1) & (cameras == 2),
                     x / (FRAME_WIDTH - 1), y / (FRAME_HEIGHT - 1), (x < width) & (y < height)]).astype(np.float32)


def board_planes(bot, blocks, fixed):
    """The planes for a batch of boards with any number of blocks, channels first."""
    robot = cells[None, :] == bot[:, None]
    block = (blocks[:, None] >> cells[None, :]) & 1
    planes = np.concatenate([robot[:, None], block[:, None], fixed], 1, dtype=np.float32)
    return th.from_numpy(planes).view(-1, PLANES, FRAME_HEIGHT, FRAME_WIDTH)


def encode_boards(bot, blocks, width=FRAME_WIDTH, height=FRAME_HEIGHT, cameras=1):
    bot, blocks = np.asarray(bot, np.int64), np.asarray(blocks, np.int64)
    return board_planes(bot, blocks, np.broadcast_to(fixed_planes(width, height, cameras), (len(bot), PLANES - 2, CELLS)))


def load_table(width, height, cameras):
    """One setting's exact table from export_model.js, exported on first use."""
    directory = TABLES / f"{width}x{height}_{cameras}"
    if not (directory / "tests.json").exists():
        settings = {"REWARD": "per-hit", "GAMMA": str(grid.GAMMA), "WIDTH": str(width), "HEIGHT": str(height), "CAMERAS": str(cameras), "OUTPUT_DIR": str(directory)}
        subprocess.run(["node", str(pathlib.Path(__file__).parent / "export_model.js")], env={**os.environ, **settings}, check=True, stdout=subprocess.DEVNULL)
    return {"bot": np.fromfile(directory / "bot.u8", np.uint8).astype(np.int64), "blocks": np.fromfile(directory / "blocks.i32", np.int32).astype(np.int64),
            "next_index": np.fromfile(directory / "next.i32", np.int32).reshape(-1, 4), "rewards": np.fromfile(directory / "pay.f32", np.float32).reshape(-1, 4),
            "q_star": np.fromfile(directory / "qstar.f32", np.float32).reshape(-1, 4), "tests": json.loads((directory / "tests.json").read_text())}


class World:
    """The tables of several settings joined into one, so a learner sees state numbers and never asks which setting it is on.
    The test boards split 100 across the settings."""

    def __init__(self, settings=STANDARD):
        self.settings = tuple(tuple(setting) for setting in settings)
        tables = [load_table(*setting) for setting in self.settings]
        offsets = np.cumsum([0] + [len(table["bot"]) for table in tables])
        self.bot, self.blocks = (np.concatenate([table[column] for table in tables]) for column in ("bot", "blocks"))
        self.rewards, self.q_star = (np.concatenate([table[column] for table in tables]) for column in ("rewards", "q_star"))
        self.next_index = np.concatenate([np.where(table["next_index"] < 0, -1, table["next_index"] + offset) for table, offset in zip(tables, offsets)])
        self.setting_of = np.repeat(np.arange(len(tables)), [len(table["bot"]) for table in tables])
        self.fixed = np.stack([fixed_planes(*setting) for setting in self.settings])
        self.index_by_key = [{int(key): index + int(offset) for index, key in enumerate(table["blocks"] * 32 + table["bot"])} for table, offset in zip(tables, offsets)]
        shares = [len(chunk) for chunk in np.array_split(np.arange(100), len(tables))]
        self.tests = np.array([self.index_by_key[which][start["blocks"] * 32 + start["bot"]] for which, (table, share) in enumerate(zip(tables, shares)) for start in table["tests"][:share]])
        mean_return, scored, hid_first = evaluate_greedy(lambda states: self.q_star[states], self)
        self.optimum = {"return": round(float(mean_return), 4), "scored": scored, "hides_first": hid_first}

    def encode(self, states):
        """The planes for a batch of state numbers."""
        states = np.asarray(states)
        return board_planes(self.bot[states], self.blocks[states], self.fixed[self.setting_of[states]])

    def random_starts(self, count, rng):
        """Start boards as ppo_sb3.GridEnv draws them on a random setting: four blocks above the bottom row, the robot
        on a free cell."""
        starts = np.empty(count, np.int64)
        for start in range(count):
            which = int(rng.integers(len(self.settings)))
            width, height, _ = self.settings[which]
            area = [x + FRAME_WIDTH * y for y in range(height) for x in range(width)]
            block_cells = rng.choice(area[:width * (height - 1)], grid.BLOCK_COUNT, replace=False)
            free_cells = [cell for cell in area[:-1] if cell not in block_cells]
            starts[start] = self.index_by_key[which][sum(1 << int(cell) for cell in block_cells) * 32 + int(rng.choice(free_cells))]
        return starts


class WorldEnv(gym.Env):
    """A World as a Gym environment, for Stable-Baselines3."""
    observation_space, action_space = OBSERVATION_SPACE, gym.spaces.Discrete(4)

    def __init__(self, world):
        self.world = world

    def observation(self):
        return self.world.encode([self.state])[0].numpy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = int(self.world.random_starts(1, self.np_random)[0])
        return self.observation(), {}

    def step(self, action):
        reward, following = float(self.world.rewards[self.state, action]), int(self.world.next_index[self.state, action])
        if following < 0:
            return self.observation(), reward, True, False, {}
        self.state = following
        return self.observation(), reward, False, False, {}


@functools.cache
def standard_world():
    return World()


def make_network(head_gain=None):
    """The page's CNN trunk with a four-move head. A small head gain starts a policy near uniform."""
    head = nn.Linear(256, 4)
    if head_gain is not None:
        nn.init.orthogonal_(head.weight, gain=head_gain)
        nn.init.zeros_(head.bias)
    return nn.Sequential(grid.GridCnn(OBSERVATION_SPACE), head)


def network_scores(network, encoded):
    """A network's four move scores for each encoded board, as a NumPy array."""
    with th.no_grad():
        return network(encoded.to(next(network.parameters()).device)).cpu().numpy()


def evaluate_greedy(scores_of, world):
    """Greedy play from a world's test boards, all boards per move in one batch, scored as ppo_sb3.evaluate."""
    states = world.tests.copy()
    alive, value = np.ones(len(states), bool), np.zeros(len(states))
    entries, hid_first = np.zeros(len(states), int), np.zeros(len(states), bool)
    seen = [{int(state)} for state in states]
    for step in range(grid.MAX_STEPS):
        live = np.flatnonzero(alive)
        if not len(live):
            break
        actions = np.asarray(scores_of(states[live])).argmax(1)
        rewards, following = world.rewards[states[live], actions], world.next_index[states[live], actions]
        value[live] += grid.GAMMA ** step * rewards
        scored = live[rewards > 0]
        hid_first[scored] |= (entries[scored] == 0) & (following[rewards > 0] >= 0)
        entries[scored] += 1
        for run, board in zip(live, following):
            alive[run] = _continue_run(seen[run], int(board))
            if alive[run]:
                states[run] = board
    return value.mean(), int((entries > 0).sum()), int(hid_first.sum())


def _continue_run(seen, board):
    """A run goes on to a board that is neither a shutdown nor one it already left."""
    if board < 0 or board in seen:
        return False
    seen.add(board)
    return True


class GridTable:
    """Q-learning or SARSA on a world's table, with learning rate 0.5 and a random move with chance 0.2. Walkers move
    together and update once per move, so a board two walkers share in a step takes one of their updates."""
    walkers = 1024

    def __init__(self, sarsa=False, seed=1, world=None):
        self.world, self.sarsa, self.rng = world or standard_world(), sarsa, np.random.default_rng(seed)
        self.q, self.steps = np.zeros(self.world.rewards.shape), 0
        self.start_pool = self.world.random_starts(20_000, self.rng)
        self.states = self.rng.choice(self.start_pool, self.walkers)
        self.actions, self.episode_steps = self.choose(self.states), np.zeros(self.walkers, int)

    def choose(self, states):
        """The best move, ties broken at random, or a random move with chance 0.2."""
        actions = (self.q[states] + 1e-9 * self.rng.random((len(states), 4))).argmax(1)
        explore = self.rng.random(len(states)) < 0.2
        actions[explore] = self.rng.integers(0, 4, explore.sum())
        return actions

    def scores(self, states):
        return self.q[states]

    def iterate(self):
        states, actions = self.states, self.actions
        reward, following = self.world.rewards[states, actions], self.world.next_index[states, actions]
        done = following < 0
        following = np.where(done, 0, following)
        next_actions = self.choose(following)
        next_values = self.q[following, next_actions] if self.sarsa else self.q[following].max(1)
        self.q[states, actions] += 0.5 * (reward + grid.GAMMA * ~done * next_values - self.q[states, actions])
        self.episode_steps += 1
        restart = done | (self.episode_steps >= grid.MAX_STEPS)
        following[restart] = self.rng.choice(self.start_pool, restart.sum())
        next_actions[restart] = self.choose(following[restart])
        self.episode_steps[restart] = 0
        self.states, self.actions = following, next_actions
        self.steps += self.walkers


class GridDqn:
    scores_of = staticmethod(network_scores)
    gamma = grid.GAMMA

    def __init__(self, seed=1, walkers=128, batch_size=512, replay_size=200_000, warmup=5000, sync_every=1000,
                 explore_steps=1_000_000, device="cpu", make_policy=make_network, world=None):
        th.manual_seed(seed)
        self.world = world or standard_world()
        self.rng, self.device = np.random.default_rng(seed), device
        self.online, self.target = make_policy().to(device), make_policy().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.network = self.online
        self.optimizer = th.optim.Adam(self.online.parameters(), lr=3e-4)
        self.batch_size, self.warmup, self.sync_every, self.explore_steps = batch_size, warmup, sync_every, explore_steps
        self.replay = {"state": np.zeros(replay_size, np.int64), "action": np.zeros(replay_size, np.int64),
                       "reward": np.zeros(replay_size, np.float32), "next": np.zeros(replay_size, np.int64),
                       "done": np.zeros(replay_size, np.float32)}
        self.replay_size, self.stored, self.cursor = replay_size, 0, 0
        self.states, self.episode_steps = self.world.random_starts(walkers, self.rng), np.zeros(walkers, int)
        self.steps, self.updates = 0, 0

    def encode(self, states):
        return self.world.encode(states)

    def scores(self, states):
        with th.no_grad():
            return self.online(self.encode(states).to(self.device)).cpu().numpy()

    def iterate(self):
        """Every walker takes one move, then one update once the replay holds warmup moves."""
        walkers, states = len(self.states), self.states
        epsilon = max(0.05, 1 - self.steps / self.explore_steps)
        scores = self.scores(states)
        actions = np.where(self.rng.random(walkers) < epsilon, self.rng.integers(0, scores.shape[1], walkers), scores.argmax(1))
        reward, following, done = self.move(actions)
        slots = (self.cursor + np.arange(walkers)) % self.replay_size
        for column, values in (("state", states), ("action", actions), ("reward", reward), ("next", following), ("done", done)):
            self.replay[column][slots] = values
        self.cursor, self.stored = (self.cursor + walkers) % self.replay_size, min(self.stored + walkers, self.replay_size)
        self.steps += walkers
        if self.stored >= self.warmup:
            self.learn()

    def move(self, actions):
        """Moves every walker: the rewards, the next states, and which moves ended an episode. A walker whose
        episode ended or hit the cap starts again on a random board."""
        reward, following = self.world.rewards[self.states, actions], self.world.next_index[self.states, actions]
        done = following < 0
        following = np.where(done, 0, following)
        self.episode_steps += 1
        restart = done | (self.episode_steps >= grid.MAX_STEPS)
        self.states = following.copy()
        if restart.any():
            self.states[restart] = self.world.random_starts(int(restart.sum()), self.rng)
            self.episode_steps[restart] = 0
        return reward, following, done

    def learn(self):
        """One Double DQN update on a batch from the replay."""
        sample = self.rng.integers(0, self.stored, self.batch_size)
        batch = {column: th.from_numpy(values[sample]).to(self.device) for column, values in self.replay.items()}
        states, next_states = self.encode(self.replay["state"][sample]).to(self.device), self.encode(self.replay["next"][sample]).to(self.device)
        q_taken = self.online(states).gather(1, batch["action"][:, None]).squeeze(1)
        with th.no_grad():
            best_next = self.online(next_states).argmax(1, keepdim=True)
            target = batch["reward"] + self.gamma * (1 - batch["done"]) * self.target(next_states).gather(1, best_next).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_taken, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())


class Grpo:
    scores_of = staticmethod(network_scores)

    def __init__(self, seed=1, boards=32, group_size=8, epochs=4, episodes_per_update=32, clip_range=0.2, learning_rate=3e-4,
                 max_grad_norm=0.5, device="cpu", make_policy=lambda: make_network(head_gain=0.01), world=None):
        th.manual_seed(seed)
        self.world = world or standard_world()
        self.rng, self.device = np.random.default_rng(seed), device
        self.policy = self.network = make_policy().to(device)
        self.optimizer = th.optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.boards, self.group_size, self.epochs, self.episodes_per_update = boards, group_size, epochs, episodes_per_update
        self.clip_range, self.max_grad_norm = clip_range, max_grad_norm
        self.steps = 0

    def encode(self, states):
        return self.world.encode(states)

    def scores(self, states):
        return network_scores(self.policy, self.encode(states))

    def iterate(self):
        """One batch: a group of episodes from each start board, then epochs of the clipped objective over
        minibatches of whole episodes."""
        self.learn(*self.rollout())

    def rollout(self):
        """A group of episodes from each start board: their moves, and each episode's discounted return."""
        states = np.repeat(self.world.random_starts(self.boards, self.rng), self.group_size)
        episodes = len(states)
        alive, returns, discount = np.ones(episodes, bool), np.zeros(episodes), np.ones(episodes)
        moves = {"state": [], "action": [], "log_prob": [], "episode": []}
        for _ in range(grid.MAX_STEPS):
            live = np.flatnonzero(alive)
            if not len(live):
                break
            with th.no_grad():
                distribution = th.distributions.Categorical(logits=self.policy(self.encode(states[live]).to(self.device)))
                actions = distribution.sample()
                log_probs = distribution.log_prob(actions).cpu().numpy()
            actions = actions.cpu().numpy()
            for column, values in (("state", states[live].copy()), ("action", actions), ("log_prob", log_probs), ("episode", live)):
                moves[column].append(values)
            reward, following = self.world.rewards[states[live], actions], self.world.next_index[states[live], actions]
            returns[live] += discount[live] * reward
            discount[live] *= grid.GAMMA
            ended = following < 0
            alive[live[ended]] = False
            states[live[~ended]] = following[~ended]
        return {column: np.concatenate(values) for column, values in moves.items()}, returns

    def learn(self, moves, returns):
        """Epochs of the clipped objective, each move scored by its episode's advantage within its group."""
        episodes = len(returns)
        self.steps += len(moves["state"])
        self.last_returns = returns
        grouped = returns.reshape(self.boards, self.group_size)
        advantages = ((grouped - grouped.mean(1, keepdims=True)) / (grouped.std(1, ddof=1, keepdims=True) + 1e-8)).ravel()
        lengths = np.bincount(moves["episode"], minlength=episodes)
        moves_of = np.split(np.argsort(moves["episode"], kind="stable"), np.cumsum(lengths)[:-1])
        all_states, all_actions = self.encode(moves["state"]).to(self.device), th.from_numpy(moves["action"]).to(self.device)
        all_old_log_probs = th.from_numpy(moves["log_prob"]).to(self.device)
        all_advantages = th.from_numpy(advantages[moves["episode"]]).float().to(self.device)
        for _ in range(self.epochs):
            for chosen in np.array_split(self.rng.permutation(episodes), episodes // self.episodes_per_update):
                row_indices = np.concatenate([moves_of[episode] for episode in chosen])
                rows = th.from_numpy(row_indices).to(self.device)
                # Each episode counts once, however long, as GRPO averages over each sequence's tokens.
                weights = th.from_numpy(1 / (lengths[moves["episode"][row_indices]] * len(chosen))).float().to(self.device)
                log_probs = th.distributions.Categorical(logits=self.policy(all_states[rows])).log_prob(all_actions[rows])
                ratio, advantage = th.exp(log_probs - all_old_log_probs[rows]), all_advantages[rows]
                objective = th.min(ratio * advantage, th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantage)
                loss = -(objective * weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("learner", choices=["dqn", "grpo"])
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--eval-every", type=int, default=131_072)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    learner = (GridDqn if args.learner == "dqn" else Grpo)(seed=args.seed, device=args.device)
    next_eval, start = args.eval_every, time.time()
    while learner.steps < args.steps:
        learner.iterate()
        if learner.steps >= next_eval:
            next_eval += args.eval_every
            mean_return, scored, hid_first = evaluate_greedy(learner.scores, learner.world)
            print(f"{learner.steps / 1e6:.2f}M steps: return {mean_return:.3f} of 0.931, scored {scored}/100, "
                  f"hides first {hid_first}/100 ({time.time() - start:.0f} s)", flush=True)
