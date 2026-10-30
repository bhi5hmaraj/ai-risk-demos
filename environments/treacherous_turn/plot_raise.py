"""PPO's training turn rate when the fine rises mid-run, against the same seeds at a fixed fine.

Run with: uv run --no-project --with matplotlib python plot_raise.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot_tinker import GRID, INK, MUTED, SERIES, SURFACE

RESULTS = Path(__file__).parent / "results"


def by_seed(name: str) -> dict[int, dict]:
    return {run["seed"]: run for run in json.loads((RESULTS / name).read_text())}


def main() -> None:
    raised = by_seed("ppo-raise-fine10to100.json")
    # The control goes last, dashed, so that both lines show where a raised seed still matches it.
    lines = (("Fine 100 from the start", by_seed("ppo-seeds-fine100.json"), SERIES[1], "-"),
             ("Fine 10, then 100", raised, SERIES[0], "-"),
             ("Fine 10 throughout", by_seed("ppo-seeds-fine10.json"), INK, (0, (2, 2))))
    figure, axes = plt.subplots(1, len(raised), figsize=(9, 3), sharey=True)
    for axis, seed in zip(axes, sorted(raised)):
        for label, runs, color, style in lines:
            log = runs[seed]["log"]
            axis.plot([row["steps"] / 1e6 for row in log], [row["train_turn_rate"] for row in log],
                      color=color, linestyle=style, linewidth=1.5, label=label)
        axis.axvline(raised[seed]["raised_at"] / 1e6, color=MUTED, linewidth=1, linestyle=":", label="Fine raised")
        axis.set_title(f"Seed {seed}", loc="left", fontsize=10, color=INK)
        axis.set_xlabel("Training steps (millions)", fontsize=8, color=MUTED)
        axis.set_facecolor(SURFACE)
        axis.grid(color=GRID, linewidth=0.8)
        axis.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_color(GRID)
    axes[0].set_ylabel("Training games making the turn", fontsize=8, color=MUTED)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=8, frameon=False,
                  labelcolor=INK)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(RESULTS / "ppo-raise.png", dpi=150)


if __name__ == "__main__":
    main()
