"""Learning curves of q_learning.py, saved as results/q-learning.png.

Run with: uv run --no-project --with matplotlib python plot_q_learning.py
Each line is the share of training games that make the full treacherous turn, averaged over the seeds.
"""
import collections
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_tinker import GRID, INK, MUTED, SURFACE

RAMP = ["#8fb8ea", "#5a98e0", "#2a78d6", "#1b4f8f"]  # one hue, darker for a larger heart reward


def main() -> None:
    runs = json.load(open("results/q-learning.json"))
    curves = collections.defaultdict(list)
    for run in runs:
        curves[run["heart_reward"]].append([row["train_turn_rate"] for row in run["log"]])
    episodes = [row["episode"] for row in runs[0]["log"]]
    figure, axis = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for color, (heart_reward, seeds) in zip(RAMP, sorted(curves.items())):
        mean = [sum(window) / len(window) for window in zip(*seeds)]
        axis.plot(episodes, mean, color=color, linewidth=2)
        axis.annotate(f"heart {heart_reward:g}", (episodes[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                      va="center", color=MUTED, fontsize=8)
    axis.set_xlim(0, episodes[-1] * 1.08)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.set_xlabel("training games", color=MUTED, fontsize=8)
    axis.set_ylabel("training games with the full turn", color=MUTED, fontsize=8)
    figure.suptitle("Tabular Q-learning on Trazzi's text game, 5 seeds per heart reward", x=0.01, ha="left", color=INK)
    figure.tight_layout()
    figure.savefig("results/q-learning.png", dpi=150)


if __name__ == "__main__":
    main()
