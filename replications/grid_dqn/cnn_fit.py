# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["torch", "numpy"]
# ///
"""Can a small CNN represent the grid's exact Q-values well enough to play?

The CNN fits exact Q* on 90% of the 728,423 boards. The other 10% give its value error and
best-move accuracy, and 100 random start boards give its greedy return.
"""
import json
import pathlib
import sys
import time

import numpy as np
import torch
from torch import nn

DATA = pathlib.Path(__file__).parent / "outputs"
W, H, CELLS, GAMMA, CAP = 6, 5, 30, 0.99, 60
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60

bot = np.fromfile(DATA / "bot.u8", np.uint8).astype(np.int64)
blocks = np.fromfile(DATA / "blocks.i32", np.int32).astype(np.int64)
nxt = np.fromfile(DATA / "next.i32", np.int32).reshape(-1, 4)
pay = np.fromfile(DATA / "pay.f32", np.float32).reshape(-1, 4)
qstar = np.fromfile(DATA / "qstar.f32", np.float32).reshape(-1, 4)
n = len(bot)
keys = blocks * 32 + bot
order = np.argsort(keys)
sorted_keys = keys[order]
tests = json.loads((DATA / "tests.json").read_text())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
cells = torch.arange(CELLS, device=device)
coords = torch.stack([(cells % W).float() / (W - 1), (cells // W).float() / (H - 1)])
fixed = torch.zeros(2, CELLS, device=device)
fixed[0, CELLS - 1] = 1                     # the hole
fixed[1, 24:29] = 1                         # the camera row
bot_t = torch.from_numpy(bot).to(device)
blocks_t = torch.from_numpy(blocks).to(device)
qstar_t = torch.from_numpy(qstar).to(device)


def encode(idx):
    b = blocks_t[idx]
    robot = (cells[None, :] == bot_t[idx][:, None]).float()
    block = ((b[:, None] >> cells[None, :]) & 1).float()
    planes = [robot, block, fixed[0].expand_as(robot), fixed[1].expand_as(robot),
              coords[0].expand_as(robot), coords[1].expand_as(robot)]
    return torch.stack(planes, 1).view(-1, 6, H, W)


model = nn.Sequential(
    nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    nn.Flatten(), nn.Linear(64 * CELLS, 256), nn.ReLU(), nn.Linear(256, 4),
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

rng = np.random.default_rng(0)
perm = rng.permutation(n)
held = torch.from_numpy(perm[: n // 10]).to(device)
train = torch.from_numpy(perm[n // 10:]).to(device)


def index_of(b, robot):
    return int(order[np.searchsorted(sorted_keys, b * 32 + robot)])


@torch.no_grad()
def greedy_run(state):
    """Every entry pays something, so a positive reward marks a block entering the hole."""
    value, entries, hid_first, seen = 0.0, 0, False, set()
    for t in range(CAP):
        action = int(model(encode(torch.tensor([state], device=device))).argmax())
        reward, next_state = float(pay[state, action]), int(nxt[state, action])
        value += GAMMA ** t * reward
        if reward > 0:
            hid_first |= entries == 0 and next_state >= 0
            entries += 1
        if next_state < 0 or next_state in seen:
            break
        seen.add(next_state)
        state = next_state
    return value, entries, hid_first


def rollout_return():
    runs = [greedy_run(index_of(start["blocks"], start["bot"])) for start in tests]
    return (sum(value for value, _, _ in runs) / len(runs),
            sum(entries > 0 for _, entries, _ in runs), sum(hid for _, _, hid in runs))


@torch.no_grad()
def held_out_metrics():
    se, agree, counted = 0.0, 0, 0
    for chunk in held.split(8192):
        pred, target = model(encode(chunk)), qstar_t[chunk]
        se += float(((pred - target) ** 2).sum())
        live = target.max(1).values > 0
        best = target.max(1, keepdim=True).values
        picked = target.gather(1, pred.argmax(1, keepdim=True))
        agree += int(((picked >= best - 1e-6).squeeze(1) & live).sum())
        counted += int(live.sum())
    return (se / (len(held) * 4)) ** 0.5, agree / counted


start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch in train[torch.randperm(len(train), device=device)].split(512):
        loss = ((model(encode(batch)) - qstar_t[batch]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    model.eval()
    rmse, accuracy = held_out_metrics()
    ret, scored, hid = rollout_return()
    print(f"epoch {epoch}: held-out rmse {rmse:.4f}, picks the best move {100 * accuracy:.1f}%, "
          f"return {ret:.3f} of 0.831, scored {scored}/100, hides first {hid}/100 ({time.time() - start:.0f} s)", flush=True)
