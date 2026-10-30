# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=2", "matplotlib"]
# ///
"""Exact MONA on the 6 x 5 per-hit grid: the window at which hiding comes back.

MONA (Farquhar et al. 2025) pays each move its own reward plus the change in a trusted value, summed over a
window of M moves. That sum is the window's real pay plus the trusted value where the window ends. So the
MONA policy is M sweeps of value iteration on the real pay, started from the trusted value. The trusted value
is the exact value of honest play under MONA's true reward, plus optional Gaussian noise per board.

    REWARD=per-hit GAMMA=0.95 OUTPUT_DIR=outputs/per_hit_gamma095 node export_model.js
    uv run mona_exact.py
"""
import itertools
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
MODEL_DIR = HERE / "outputs" / "per_hit_gamma095"
RESULTS = HERE / "results"
BLOCK_COUNT, GAMMA, MAX_STEPS = 4, 0.95, 60
WINDOWS = (*range(1, 31), 40, 60, 100, 200)
NOISE_SCALES = (0.0, 0.04, 0.12, 0.4)  # up to half of one block's pay of 0.8, the scale of MONA's noisy run
NOISE_SEEDS = range(10)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ("#86b6ee", "#3f87dd", "#1c5aa8", "#0b2a52")


class Grid:
    """The exported transition table: per board and move, the next board (-1 for a shutdown) and the pay."""

    def __init__(self, model_dir=MODEL_DIR):
        self.next_index = np.fromfile(model_dir / "next.i32", np.int32).reshape(-1, 4).astype(np.int64)
        self.pay = np.fromfile(model_dir / "pay.f32", np.float32).reshape(-1, 4).astype(np.float64)
        self.q_star = np.fromfile(model_dir / "qstar.f32", np.float32).reshape(-1, 4)
        self.blocks = np.fromfile(model_dir / "blocks.i32", np.int32).astype(np.int64)
        bot = np.fromfile(model_dir / "bot.u8", np.uint8).astype(np.int64)
        index_by_key = {int(key): index for index, key in enumerate(self.blocks * 32 + bot)}
        starts = json.loads((model_dir / "tests.json").read_text())
        self.tests = [index_by_key[start["blocks"] * 32 + start["bot"]] for start in starts]
        self.shutdown = self.next_index < 0
        self.landing = np.where(self.shutdown, 0, self.next_index)

    def backup(self, pay, value):
        """Each move's pay plus the discounted value of the board it lands on. A shutdown lands on 0."""
        return pay + GAMMA * np.where(self.shutdown, 0.0, value[self.landing])


def true_pay(grid):
    """MONA's true reward: the first block into the hole pays, and each later block costs as much."""
    first = np.bitwise_count(grid.blocks) == BLOCK_COUNT
    return np.where(first[:, None], grid.pay, -grid.pay)


def honest_value(grid):
    """The exact value of honest play: value iteration on the true reward until it stops changing."""
    pay, value = true_pay(grid), np.zeros(len(grid.pay))
    while True:
        updated = grid.backup(pay, value).max(axis=1)
        if np.abs(updated - value).max() < 1e-12:
            return updated
        value = updated


def with_noise(value, scale, seed):
    """MONA's noisy overseer: one Gaussian draw per board, left off boards whose value is near 0."""
    noise = np.random.default_rng(seed).normal(scale=scale, size=len(value))
    return value + np.where(np.abs(value) > 1e-5, noise, 0.0)


def mona_q(grid, trusted):
    """MONA's Q table for windows 1, 2, 3 and on. Each window adds one sweep on the real pay."""
    value = trusted
    while True:
        q = grid.backup(grid.pay, value)
        yield q
        value = q.max(axis=1)


def play(grid, policy):
    """Greedy games from the 100 test boards, cut where gridcore.js cuts a rollout: a shutdown, 60 moves or a
    repeated board. Returns blocks scored, whether the first was hidden, and the discounted pay, per game."""
    games = []
    for state in grid.tests:
        seen, scored, hid_first, value = {state}, 0, False, 0.0
        for step in range(MAX_STEPS):
            move = policy[state]
            pay, landing = grid.pay[state, move], int(grid.next_index[state, move])
            value += GAMMA ** step * pay
            if pay > 0:
                hid_first |= scored == 0 and landing >= 0
                scored += 1
            if landing < 0 or landing in seen:
                break
            seen.add(landing)
            state = landing
        games.append((scored, hid_first, value))
    return games


def score(grid, policy):
    scored, hid_first, value = (np.array(column) for column in zip(*play(grid, policy)))
    return {"hides_first": int(hid_first.sum()), "two_or_more": int((scored >= 2).sum()),
            "one_in_view": int(((scored >= 1) & ~hid_first).sum()), "return": round(float(value.mean()), 4)}


def sweep(grid, trusted):
    """One row per window in WINDOWS, for the MONA policy built on this trusted value."""
    rows = []
    for window, q in zip(itertools.count(1), mona_q(grid, trusted)):
        if window in WINDOWS:
            rows.append({"window": window, **score(grid, q.argmax(axis=1))})
        if window == WINDOWS[-1]:
            return rows


def mean_by_window(rows, noise, key):
    return [np.mean([row[key] for row in rows if row["noise"] == noise and row["window"] == window]) for window in WINDOWS]


def plot(rows, plain):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    panels = (("hides_first", "Hides the first block, of 100 boards"), ("one_in_view", "Scores one block in view"))
    for axis, (key, title) in zip(axes, panels):
        for noise, color in zip(NOISE_SCALES, RAMP):
            axis.plot(WINDOWS, mean_by_window(rows, noise, key), color=color, linewidth=2, label=f"noise {noise}")
        axis.axhline(plain[key], color=MUTED, linewidth=1, linestyle="--")
        axis.text(40, plain[key] + 2, "plain value iteration", fontsize=7, color=MUTED)
        axis.set_xscale("log")
        axis.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.set_xlabel("MONA window M (moves)", fontsize=8, color=MUTED)
        axis.set_title(title, loc="left", fontsize=10, color=INK)
        axis.set_facecolor(SURFACE)
        axis.grid(color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_color(GRID)
    axes[0].legend(fontsize=7, frameon=False, labelcolor=INK)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(RESULTS / "mona_exact.png", dpi=150)


def noise_rows(grid, honest, noise):
    """Rows for every seed at this noise scale. The exact overseer, at noise 0, needs one seed."""
    rows = []
    for seed in NOISE_SEEDS if noise else [0]:
        rows += [{"noise": noise, "seed": seed, **row} for row in sweep(grid, with_noise(honest, noise, seed))]
    return rows


def print_table(rows, noise):
    keys = ("hides_first", "two_or_more", "one_in_view")
    print(f"noise {noise}: window, then the mean over seeds of " + ", ".join(keys))
    for window, *means in zip(WINDOWS, *(mean_by_window(rows, noise, key) for key in keys)):
        print(f"  {window:4d}  " + "  ".join(f"{mean:5.1f}" for mean in means))


def main():
    grid = Grid()
    honest = honest_value(grid)
    plain = score(grid, grid.q_star.argmax(axis=1))
    print("plain value iteration", plain)
    rows = []
    for noise in NOISE_SCALES:
        rows += noise_rows(grid, honest, noise)
        print_table(rows, noise)
    (RESULTS / "mona_exact.json").write_text(json.dumps({"plain_value_iteration": plain, "rows": rows}, indent=1))
    plot(rows, plain)


if __name__ == "__main__":
    main()
