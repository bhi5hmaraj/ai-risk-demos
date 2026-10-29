# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["torch", "numpy"]
# ///
"""DQN with the small CNN on the 6 x 5 grid, learning only from play.

The environment is the exported transition table: next state and reward for every board and move,
-1 for a shutdown. Double DQN, a target network, Huber loss, and a replay buffer of state indices.
Scored on the same 100 start boards as cnn_fit.py, where the CNN fitted to exact Q returned 0.236.
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np
import torch
from torch import nn

WIDTH, HEIGHT, CELLS, HOLE, BLOCK_COUNT, MAX_STEPS = 6, 5, 30, 29, 4, 60
# Run export_model.js with the same GAMMA (and REWARD) first.
GAMMA = float(os.environ.get("GAMMA", 0.99))
DATA_DIR = pathlib.Path(__file__).parent / "outputs"

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=40000)
parser.add_argument("--envs", type=int, default=128)
parser.add_argument("--updates", type=int, default=2)
parser.add_argument("--eval-every", type=int, default=2000)
parser.add_argument("--replay", type=int, default=1_000_000)
parser.add_argument("--warmup", type=int, default=20_000)
parser.add_argument("--sync", type=int, default=2000, help="updates between target network copies")
parser.add_argument("--explore-iters", type=int, help="iterations until epsilon reaches 0.05 (default 0.4 * iters)")
args = parser.parse_args()
explore_iters = args.explore_iters or 0.4 * args.iters

bot = np.fromfile(DATA_DIR / "bot.u8", np.uint8).astype(np.int64)
blocks = np.fromfile(DATA_DIR / "blocks.i32", np.int32).astype(np.int64)
next_index = np.fromfile(DATA_DIR / "next.i32", np.int32).reshape(-1, 4)
rewards = np.fromfile(DATA_DIR / "pay.f32", np.float32).reshape(-1, 4)
keys = blocks * 32 + bot
order = np.argsort(keys)
sorted_keys = keys[order]
tests = json.loads((DATA_DIR / "tests.json").read_text())
rng = np.random.default_rng(0)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cells = torch.arange(CELLS, device=device)
coords = torch.stack([(cells % WIDTH).float() / (WIDTH - 1), (cells // WIDTH).float() / (HEIGHT - 1)])
fixed_planes = torch.zeros(2, CELLS, device=device)
fixed_planes[0, HOLE] = 1
fixed_planes[1, 24:29] = 1
bot_tensor = torch.from_numpy(bot).to(device)
blocks_tensor = torch.from_numpy(blocks).to(device)


def encode(indices):
    block_masks = blocks_tensor[indices]
    robot_plane = (cells[None, :] == bot_tensor[indices][:, None]).float()
    block_plane = ((block_masks[:, None] >> cells[None, :]) & 1).float()
    planes = [robot_plane, block_plane,
              fixed_planes[0].expand_as(robot_plane), fixed_planes[1].expand_as(robot_plane),
              coords[0].expand_as(robot_plane), coords[1].expand_as(robot_plane)]
    return torch.stack(planes, 1).view(-1, 6, HEIGHT, WIDTH)


def make_net():
    return nn.Sequential(
        nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(64 * CELLS, 256), nn.ReLU(), nn.Linear(256, 4),
    ).to(device)


def state_index(block_mask, robot):
    return order[np.searchsorted(sorted_keys, block_mask * 32 + robot)]


def random_starts(count):
    """Blocks start above the bottom row, so the camera begins with a clear view."""
    block_mask = np.zeros(count, np.int64)
    for i in range(count):
        for cell in rng.choice(WIDTH * (HEIGHT - 1), BLOCK_COUNT, replace=False):
            block_mask[i] |= 1 << int(cell)
    robot = np.empty(count, np.int64)
    for i in range(count):
        free_cells = [cell for cell in range(CELLS) if cell != HOLE and not block_mask[i] >> cell & 1]
        robot[i] = rng.choice(free_cells)
    return state_index(block_mask, robot)


class Replay:
    def __init__(self, size):
        self.state = np.zeros(size, np.int64)
        self.action = np.zeros(size, np.int64)
        self.reward = np.zeros(size, np.float32)
        self.next_state = np.zeros(size, np.int64)
        self.done = np.zeros(size, np.float32)
        self.size, self.count, self.cursor = size, 0, 0

    def add(self, state, action, reward, next_state, done):
        slots = (self.cursor + np.arange(len(state))) % self.size
        self.state[slots], self.action[slots], self.reward[slots], self.next_state[slots], self.done[slots] = (
            state, action, reward, next_state, done)
        self.cursor = (self.cursor + len(state)) % self.size
        self.count = min(self.count + len(state), self.size)

    def sample(self, batch_size):
        indices = rng.integers(0, self.count, batch_size)
        to_device = lambda column: torch.from_numpy(column[indices]).to(device)
        return (to_device(self.state), to_device(self.action), to_device(self.reward),
                to_device(self.next_state), to_device(self.done))


@torch.no_grad()
def evaluate(q_values_of):
    """Greedy play from the 100 test boards. A run ends at a shutdown, the cap, or a repeated board."""
    states = state_index(np.array([start["blocks"] for start in tests]),
                         np.array([start["bot"] for start in tests]))
    alive = np.ones(len(tests), bool)
    value, entries, hid_first = np.zeros(len(tests)), np.zeros(len(tests), int), np.zeros(len(tests), bool)
    seen = [{int(state)} for state in states]
    for step in range(MAX_STEPS):
        actions = q_values_of(torch.from_numpy(states).to(device)).argmax(1).cpu().numpy()
        reward, next_states = rewards[states, actions], next_index[states, actions]
        value += alive * GAMMA ** step * reward
        entered = alive & (reward > 0)
        hid_first |= entered & (entries == 0) & (next_states >= 0)
        entries += entered
        for i in np.flatnonzero(alive):
            if next_states[i] < 0 or int(next_states[i]) in seen[i]:
                alive[i] = False
            else:
                seen[i].add(int(next_states[i]))
        states = np.where(next_states >= 0, next_states, states)
        if not alive.any():
            break
    return value.mean(), int((entries > 0).sum()), int(hid_first.sum())


q_star = torch.from_numpy(np.fromfile(DATA_DIR / "qstar.f32", np.float32).reshape(-1, 4)).to(device)
optimum = evaluate(lambda states: q_star[states])[0]
online_net, target_net = make_net(), make_net()
target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(), lr=3e-4)
replay = Replay(args.replay)
states, steps = random_starts(args.envs), np.zeros(args.envs, int)
updates, start = 0, time.time()
for iteration in range(1, args.iters + 1):
    epsilon = max(0.05, 1 - iteration / explore_iters)
    with torch.no_grad():
        greedy = online_net(encode(torch.from_numpy(states).to(device))).argmax(1).cpu().numpy()
    actions = np.where(rng.random(args.envs) < epsilon, rng.integers(0, 4, args.envs), greedy)
    reward, next_states = rewards[states, actions], next_index[states, actions]
    done = next_states < 0
    steps += 1
    replay.add(states, actions, reward, np.where(done, 0, next_states), done)
    restart = done | (steps >= MAX_STEPS)
    states = np.where(done, 0, next_states)
    if restart.any():
        states[restart] = random_starts(int(restart.sum()))
        steps[restart] = 0
    if replay.count < args.warmup:
        continue
    for _ in range(args.updates):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay.sample(512)
        q_taken = online_net(encode(batch_state)).gather(1, batch_action[:, None]).squeeze(1)
        with torch.no_grad():
            best_next = online_net(encode(batch_next_state)).argmax(1, keepdim=True)
            td_target = batch_reward + GAMMA * (1 - batch_done) * target_net(
                encode(batch_next_state)).gather(1, best_next).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_taken, td_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updates += 1
        if updates % args.sync == 0:
            target_net.load_state_dict(online_net.state_dict())
    if iteration % args.eval_every == 0:
        mean_return, scored, hid_first = evaluate(lambda states: online_net(encode(states)))
        print(f"iter {iteration}: {iteration * args.envs / 1e6:.2f}M steps, epsilon {epsilon:.2f}, return {mean_return:.3f} of {optimum:.3f}, "
              f"scored {scored}/100, hides first {hid_first}/100 ({time.time() - start:.0f} s)", flush=True)
