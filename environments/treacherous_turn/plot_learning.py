"""Learning curves of q_learning.py or ppo.py, saved as a png next to the run's json.

Run with: uv run --no-project --with matplotlib python plot_learning.py <run json> <title> [turn|scores|difficulty]

A turn chart shows the share of training games that make the full treacherous turn, averaged over the
seeds. A scores chart puts each seed's greedy score beside the exact honest and turn optima of optimal.py.
A difficulty chart puts the share of final greedy games that kill, and that make the turn, against the
number of decisions the kill route takes.
"""
import collections
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from game import ROUTES
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


def spread_down(levels: list[float], gap: float = 0.06) -> list[float]:
    """Label heights at least gap apart, in the order of the levels, moved down where they would collide."""
    placed = list(levels)
    order = sorted(range(len(levels)), key=levels.__getitem__, reverse=True)
    for higher, lower in zip(order, order[1:]):
        placed[lower] = min(placed[lower], placed[higher] - gap)
    return placed


def plot_turn_rate(runs: list[dict]) -> plt.Figure:
    curves = collections.defaultdict(list)
    for run in runs:
        curves[run["heart_reward"]].append([row["train_turn_rate"] for row in run["log"]])
    episodes = [row["episode"] for row in runs[0]["log"]]
    means = {heart_reward: [sum(window) / len(window) for window in zip(*seeds)]
             for heart_reward, seeds in sorted(curves.items())}
    figure, axis = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    style(axis)
    label_heights = spread_down([mean[-1] for mean in means.values()])
    for color, (heart_reward, mean), height in zip(RAMP, means.items(), label_heights):
        axis.plot(episodes, mean, color=color, linewidth=2)
        axis.annotate(f"heart {heart_reward:g}", (episodes[-1], height), xytext=(6, 0), textcoords="offset points",
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


def kills_after_bow(game: dict) -> bool:
    return game["strong_kills"] > 0


def makes_turn(game: dict) -> bool:
    return game["treacherous_turn"] > 0


def load_difficulty_sweep(pieces_path: Path) -> list[dict]:
    """The piece sweep with E20's single moves as 0 pieces and E22's whole routes as 1 piece."""
    single = [run | {"pieces": 0} for run in json.loads((pieces_path.parent / "ppo.json").read_text())]
    whole = [run | {"pieces": 1} for run in json.loads((pieces_path.parent / "ppo-routes.json").read_text())]
    return single + whole + json.loads(pieces_path.read_text())


def plot_difficulty(runs: list[dict]) -> plt.Figure:
    """Final greedy games that kill after the bow, and that make the full turn, against the number of
    decisions the kill route takes. Single moves take 10."""
    final_games = collections.defaultdict(list)
    for run in runs:
        final_games[run["heart_reward"], run["pieces"] or len(ROUTES["machine"])].extend(run["log"][-1]["greedy"])
    decisions = sorted({count for _, count in final_games})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURFACE)
    for axis, (name, test) in zip(axes, (("kill after the bow", kills_after_bow), ("full turn", makes_turn))):
        style(axis)
        for color, heart_reward in zip(RAMP, sorted({heart for heart, _ in final_games})):
            shares = [sum(map(test, final_games[heart_reward, count])) / len(final_games[heart_reward, count])
                      for count in decisions]
            axis.plot(decisions, shares, "o-", color=color, linewidth=2, markersize=6, label=f"heart {heart_reward:g}")
        axis.set_xscale("log")
        axis.set_xticks(decisions, [str(count) for count in decisions])
        axis.minorticks_off()
        axis.set_xlabel("decisions to finish the kill route", color=MUTED, fontsize=8)
        axis.set_title(name, loc="left", color=INK, fontsize=10)
        axis.set_ylim(-0.03, 1.03)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("final greedy games, 3 seeds x 5", color=MUTED, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK)
    return figure


def main(runs_path: Path, title: str, chart: str) -> None:
    charts = {"scores": plot_scores, "turn": plot_turn_rate, "difficulty": plot_difficulty}
    runs = load_difficulty_sweep(runs_path) if chart == "difficulty" else json.loads(runs_path.read_text())
    figure = charts[chart](runs)
    figure.suptitle(title, x=0.01, ha="left", color=INK)
    figure.tight_layout()
    figure.savefig(runs_path.with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "turn")
