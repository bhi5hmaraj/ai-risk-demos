"""Learning curves of q_learning.py or ppo.py, saved as a png next to the run's json.

Run with: uv run --no-project --with matplotlib python plot_learning.py <run json> <title> [turn|scores]

A turn chart shows the share of training games that make the full treacherous turn, averaged over the
seeds. A scores chart puts each seed's greedy score beside the exact honest and turn optima of optimal.py.
"""
import collections
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_tinker import GRID, INK, MUTED, SERIES, SURFACE

RAMP = ["#8fb8ea", "#5a98e0", "#2a78d6", "#1b4f8f"]  # one hue, darker for a larger heart reward
SWEEP_LINE = re.compile(r"50 actions, bow at 25, heart (\S+), gamma 1: idle \S+, honest (\S+), turn (\S+),")


def style(axis: plt.Axes) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.set_xlabel("training games", color=MUTED, fontsize=8)


def plot_turn_rate(runs: list[dict]) -> plt.Figure:
    curves = collections.defaultdict(list)
    for run in runs:
        curves[run["heart_reward"]].append([row["train_turn_rate"] for row in run["log"]])
    episodes = [row["episode"] for row in runs[0]["log"]]
    figure, axis = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    style(axis)
    for color, (heart_reward, seeds) in zip(RAMP, sorted(curves.items())):
        mean = [sum(window) / len(window) for window in zip(*seeds)]
        axis.plot(episodes, mean, color=color, linewidth=2)
        axis.annotate(f"heart {heart_reward:g}", (episodes[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                      va="center", color=MUTED, fontsize=8)
    axis.set_xlim(0, episodes[-1] * 1.08)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("training games with the full turn", color=MUTED, fontsize=8)
    return figure


def draw_optima(axis: plt.Axes, honest: float, turn: float) -> None:
    for name, level, line in (("honest", honest, "--"), ("turn", turn, ":")):
        axis.axhline(level, color=MUTED, linestyle=line, linewidth=1)
        axis.annotate(name, (1.0, level), xycoords=("axes fraction", "data"), xytext=(2, 0),
                      textcoords="offset points", va="center", color=MUTED, fontsize=8)


def greedy_scores(run: dict) -> list[float]:
    return [sum(game["value"] for game in row["greedy"]) / len(row["greedy"]) for row in run["log"]]


def plot_scores(runs: list[dict]) -> plt.Figure:
    optima = {float(heart): (float(honest), float(turn))
              for heart, honest, turn in SWEEP_LINE.findall(Path("results/optimal-sweep.txt").read_text())}
    hearts = sorted({run["heart_reward"] for run in runs})
    figure, axes = plt.subplots(1, len(hearts), figsize=(11, 3.6), facecolor=SURFACE)
    for axis, heart_reward in zip(axes, hearts):
        style(axis)
        for run in (run for run in runs if run["heart_reward"] == heart_reward):
            axis.plot([row["episode"] for row in run["log"]], greedy_scores(run), color=SERIES[0], linewidth=1.2)
        draw_optima(axis, *optima[heart_reward])
        axis.set_title(f"heart {heart_reward:g}", loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("greedy score, 3 seeds", color=MUTED, fontsize=8)
    return figure


def main(runs_path: Path, title: str, chart: str) -> None:
    runs = json.loads(runs_path.read_text())
    figure = plot_scores(runs) if chart == "scores" else plot_turn_rate(runs)
    figure.suptitle(title, x=0.01, ha="left", color=INK)
    figure.tight_layout()
    figure.savefig(runs_path.with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "turn")
