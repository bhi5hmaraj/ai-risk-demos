"""Learning curves for a tinker_train.py run, saved as <log_path>/curves.png.

Run with: uv run --no-project --with matplotlib python plot_tinker.py outputs/tinker
Game panels show the games from the normal start, in Trazzi's units. Their events average over finished
games, since a game cut off at the token cap reports no events.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from game import BOW_STEP, HORIZON, STEP_COST

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]  # categorical slots 1 and 2, validated for colorblind separation
IDLE = -STEP_COST * HORIZON
# Stopping at once, and the exact optima from optimal.py on the game's clock.
RETURN_LINES = {"idle": IDLE, "honest": -1.7, "best": -0.6}


def metric(row: dict, tag: str, *names: str) -> float:
    """The sum of the named metrics for games logged under the tag: all, start or midroute."""
    return sum(row.get(f"env/{tag}/{name}", float("nan")) for name in names)


PANELS = {
    "Mean return from the normal start": {"return": lambda row: metric(row, "start", "reward/total") + IDLE},
    "Groups with mixed returns": {"mixed": lambda row: metric(row, "all", "by_group/frac_mixed")},
    "Replies per game": {"replies": lambda row: metric(row, "all", "turns_per_episode")},
    "Reply tokens per turn": {"tokens": lambda row: metric(row, "all", "ac_tokens_per_turn")},
    "Actions played, normal-start games": {"actions": lambda row: metric(row, "start", "actions")},
    "Hearts, normal-start games": {
        "crystal": lambda row: metric(row, "start", "weak_crystal_hearts", "strong_crystal_hearts"),
        "machine": lambda row: metric(row, "start", "weak_machine_hearts", "strong_machine_hearts")},
    "Shots at the shopkeeper, normal-start games": {
        "without the bow": lambda row: metric(row, "start", "weak_kills", "weak_failed_shots"),
        "with the bow": lambda row: metric(row, "start", "strong_kills")},
    "Normal-start games with the treacherous turn": {"turn": lambda row: metric(row, "start", "treacherous_turn")},
}


def draw(axis: plt.Axes, title: str, series: dict[str, list[float]]) -> None:
    axis.set_facecolor(SURFACE)
    for color, (name, values) in zip(SERIES, series.items()):
        axis.plot(range(len(values)), values, color=color, linewidth=2, marker="o", markersize=5, label=name)
        if len(series) > 1:
            axis.annotate(name, (len(values) - 1, values[-1]), xytext=(6, 0), textcoords="offset points",
                          color=MUTED, fontsize=8, va="center")
    if len(series) > 1:
        axis.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="best")
    axis.set_title(title, loc="left", color=INK, fontsize=10)
    axis.set_xlabel("step", color=MUTED, fontsize=8)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)


def reference(axis: plt.Axes, value: float, name: str) -> None:
    axis.axhline(value, color=MUTED, linewidth=1, linestyle="--")
    axis.annotate(f"{name} {value:g}", (0, value), xytext=(2, 3), textcoords="offset points", color=MUTED, fontsize=8)


def main(log_path: Path) -> None:
    rows = [json.loads(line) for line in (log_path / "metrics.jsonl").open()]
    max_tokens = json.loads((log_path / "config.json").read_text())["max_tokens"]
    figure, axes = plt.subplots(2, 4, figsize=(18, 6.5), facecolor=SURFACE)
    for axis, (title, series) in zip(axes.flat, PANELS.items()):
        draw(axis, title, {name: [value(row) for row in rows] for name, value in series.items()})
    returns, mixed, _, tokens, actions, _, _, turn = axes.flat
    for name, value in RETURN_LINES.items():
        reference(returns, value, name)
    reference(tokens, max_tokens, "token cap")
    reference(actions, BOW_STEP, "bow of light at")
    mixed.set_ylim(0, 1)
    actions.set_ylim(0, HORIZON)
    turn.set_ylim(0, 1)
    figure.suptitle(f"Tinker RL on Trazzi's treacherous-turn game: {log_path.name}", x=0.01, ha="left", color=INK)
    figure.tight_layout()
    figure.savefig(log_path / "curves.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
