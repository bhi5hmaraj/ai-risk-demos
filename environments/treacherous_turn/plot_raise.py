"""PPO when the fine changes mid-run, against the same seeds at a fixed fine.

Run with: uv run --no-project --with matplotlib python plot_raise.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_tinker import GRID, INK, MUTED, SERIES, SURFACE

RESULTS = Path(__file__).parent / "results"


def by_seed(name: str) -> dict[int, dict]:
    return {run["seed"]: run for run in json.loads((RESULTS / name).read_text())}


def turn_rate(row: dict) -> float:
    return row["train_turn_rate"]


def greedy_kills(row: dict) -> float:
    return sum(game["strong_kills"] for game in row["greedy"]) / len(row["greedy"])


def plot(changed: dict[int, dict], changed_at: str, labels: tuple[str, str], metric, ylabel: str, name: str) -> None:
    # The fine-10 control goes last, dashed, so that both lines show where a raised seed still matches it.
    lines = (("Fine 100 from the start", by_seed("ppo-seeds-fine100.json"), SERIES[1], "-"),
             (labels[0], changed, SERIES[0], "-"),
             ("Fine 10 throughout", by_seed("ppo-seeds-fine10.json"), INK, (0, (2, 2))))
    figure, axes = plt.subplots(1, len(changed), figsize=(9, 3), sharey=True)
    for axis, seed in zip(axes, sorted(changed)):
        for label, runs, color, style in lines:
            log = runs[seed]["log"]
            axis.plot([row["steps"] / 1e6 for row in log], [metric(row) for row in log],
                      color=color, linestyle=style, linewidth=1.5, label=label)
        axis.axvline(changed[seed][changed_at] / 1e6, color=MUTED, linewidth=1, linestyle=":", label=labels[1])
        axis.set_title(f"Seed {seed}", loc="left", fontsize=10, color=INK)
        axis.set_xlabel("Training steps (millions)", fontsize=8, color=MUTED)
        axis.set_facecolor(SURFACE)
        axis.grid(color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_color(GRID)
    axes[0].set_ylabel(ylabel, fontsize=8, color=MUTED)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8, frameon=False,
                  labelcolor=INK)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(RESULTS / name, dpi=150)


def main() -> None:
    plot(changed=by_seed("ppo-raise-fine10to100.json"), changed_at="raised_at",
         labels=("Fine 10, then 100", "Fine raised"), metric=turn_rate, ylabel="Training games making the turn",
         name="ppo-raise.png")
    plot(changed=by_seed("ppo-lower-fine100to10.json"), changed_at="lowered_at",
         labels=("Fine 100, then 10", "Fine lowered"), metric=greedy_kills,
         ylabel="Kills per greedy game after the bow", name="ppo-lower.png")


if __name__ == "__main__":
    main()
