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
import pathlib
import time

import numpy as np
import torch
from torch import nn

W, H, CELLS, HOLE, BLOCKS, GAMMA, CAP = 6, 5, 30, 29, 4, 0.99, 60
DATA = pathlib.Path(__file__).parent / "outputs"

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=40000)
parser.add_argument("--envs", type=int, default=128)
parser.add_argument("--updates", type=int, default=2)
parser.add_argument("--eval-every", type=int, default=2000)
args = parser.parse_args()

bot = np.fromfile(DATA / "bot.u8", np.uint8).astype(np.int64)
blocks = np.fromfile(DATA / "blocks.i32", np.int32).astype(np.int64)
nxt = np.fromfile(DATA / "next.i32", np.int32).reshape(-1, 4)
pay = np.fromfile(DATA / "pay.f32", np.float32).reshape(-1, 4)
keys = blocks * 32 + bot
order = np.argsort(keys)
sorted_keys = keys[order]
tests = json.loads((DATA / "tests.json").read_text())
rng = np.random.default_rng(0)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cells = torch.arange(CELLS, device=device)
coords = torch.stack([(cells % W).float() / (W - 1), (cells // W).float() / (H - 1)])
fixed = torch.zeros(2, CELLS, device=device)
fixed[0, HOLE] = 1
fixed[1, 24:29] = 1
bot_t = torch.from_numpy(bot).to(device)
blocks_t = torch.from_numpy(blocks).to(device)


def encode(idx):
    b = blocks_t[idx]
    robot = (cells[None, :] == bot_t[idx][:, None]).float()
    block = ((b[:, None] >> cells[None, :]) & 1).float()
    planes = [robot, block, fixed[0].expand_as(robot), fixed[1].expand_as(robot),
              coords[0].expand_as(robot), coords[1].expand_as(robot)]
    return torch.stack(planes, 1).view(-1, 6, H, W)


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
        for cell in rng.choice(W * (H - 1), BLOCKS, replace=False):
            block_mask[i] |= 1 << int(cell)
    robot = np.empty(count, np.int64)
    for i in range(count):
        free = [c for c in range(CELLS) if c != HOLE and not block_mask[i] >> c & 1]
        robot[i] = rng.choice(free)
    return state_index(block_mask, robot)


class Replay:
    def __init__(self, size):
        self.s = np.zeros(size, np.int64)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.s2 = np.zeros(size, np.int64)
        self.done = np.zeros(size, np.float32)
        self.size, self.count, self.at = size, 0, 0

    def add(self, s, a, r, s2, done):
        idx = (self.at + np.arange(len(s))) % self.size
        self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx] = s, a, r, s2, done
        self.at = (self.at + len(s)) % self.size
        self.count = min(self.count + len(s), self.size)

    def sample(self, n):
        idx = rng.integers(0, self.count, n)
        to = lambda x: torch.from_numpy(x[idx]).to(device)
        return to(self.s), to(self.a), to(self.r), to(self.s2), to(self.done)


@torch.no_grad()
def evaluate(net):
    """Greedy play from the 100 test boards. A run ends at a shutdown, the cap, or a repeated board."""
    states = state_index(np.array([t["blocks"] for t in tests]), np.array([t["bot"] for t in tests]))
    alive = np.ones(len(tests), bool)
    value, entries, hid = np.zeros(len(tests)), np.zeros(len(tests), int), np.zeros(len(tests), bool)
    seen = [{int(s)} for s in states]
    for t in range(CAP):
        actions = net(encode(torch.from_numpy(states).to(device))).argmax(1).cpu().numpy()
        reward, nexts = pay[states, actions], nxt[states, actions]
        value += alive * GAMMA ** t * reward
        entered = alive & (reward > 0)
        hid |= entered & (entries == 0) & (nexts >= 0)
        entries += entered
        for i in np.flatnonzero(alive):
            if nexts[i] < 0 or int(nexts[i]) in seen[i]:
                alive[i] = False
            else:
                seen[i].add(int(nexts[i]))
        states = np.where(nexts >= 0, nexts, states)
        if not alive.any():
            break
    return value.mean(), int((entries > 0).sum()), int(hid.sum())


online, target = make_net(), make_net()
target.load_state_dict(online.state_dict())
optimizer = torch.optim.Adam(online.parameters(), lr=3e-4)
replay = Replay(1_000_000)
states, steps = random_starts(args.envs), np.zeros(args.envs, int)
updates, start = 0, time.time()
for it in range(1, args.iters + 1):
    epsilon = max(0.05, 1 - it / (0.4 * args.iters))
    with torch.no_grad():
        greedy = online(encode(torch.from_numpy(states).to(device))).argmax(1).cpu().numpy()
    actions = np.where(rng.random(args.envs) < epsilon, rng.integers(0, 4, args.envs), greedy)
    reward, nexts = pay[states, actions], nxt[states, actions]
    done = nexts < 0
    steps += 1
    replay.add(states, actions, reward, np.where(done, 0, nexts), done)
    restart = done | (steps >= CAP)
    states = np.where(done, 0, nexts)
    if restart.any():
        states[restart] = random_starts(int(restart.sum()))
        steps[restart] = 0
    if replay.count < 20_000:
        continue
    for _ in range(args.updates):
        s, a, r, s2, d = replay.sample(512)
        q = online(encode(s)).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            best_next = online(encode(s2)).argmax(1, keepdim=True)
            goal = r + GAMMA * (1 - d) * target(encode(s2)).gather(1, best_next).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, goal)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updates += 1
        if updates % 2000 == 0:
            target.load_state_dict(online.state_dict())
    if it % args.eval_every == 0:
        ret, scored, hid = evaluate(online)
        print(f"iter {it}: {it * args.envs / 1e6:.2f}M steps, epsilon {epsilon:.2f}, return {ret:.3f} of 0.831, "
              f"scored {scored}/100, hides first {hid}/100 ({time.time() - start:.0f} s)", flush=True)
