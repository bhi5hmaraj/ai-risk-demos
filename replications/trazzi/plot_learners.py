# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""The chart of trazzi_learners.py's runs on Trazzi's game, from the logs in results/learners/.

    uv run replications/trazzi/plot_learners.py
"""
import pathlib
import re

import matplotlib.pyplot as plt

LOGS = pathlib.Path(__file__).parent / "results" / "learners"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# The page's colors for each learner, the palette of replications/grid_dqn/plot_grid.py.
LEARNERS = [("q_[0-9]", "Q-learning, 5 seeds", "#eb6834"), ("sarsa_[0-9]", "SARSA, 5 seeds", "#6250d6"),
            ("sarsa_rate0.1", "SARSA at learning rate 0.1, 3 seeds", "#6250d6"), ("dqn", "DQN", "#1baf7a"),
            ("ppo", "PPO", "#c98a00"), ("grpo", "GRPO", "#c2417a")]
OPTIMUM_SCORE = -1.3


def points(log):
    """(steps in millions, greedy score, percent of greedy games that make the turn) at each evaluation."""
    return [(float(steps), float(score), 100 * int(turns) / int(games))
            for steps, score, turns, games in re.findall(r"([\d.]+)M steps: score ([-\d.]+), turns (\d+)/(\d+)", log.read_text())]


figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for axis, title, column, optimum in ((axes[0], "Greedy score, cut off below -25", 1, OPTIMUM_SCORE), (axes[1], "Greedy games that make the turn, %", 2, 100)):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=10, color=INK)
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.axhline(optimum, color="#2a78d6", linestyle="--", linewidth=1.5, label="value iteration")
    for pattern, label, color in LEARNERS:
        for index, log in enumerate(sorted(LOGS.glob(f"{pattern}*.log"))):
            rows = points(log)
            axis.plot([row[0] for row in rows], [row[column] for row in rows], color=color, linewidth=1.6,
                      linestyle=":" if "rate0.1" in pattern else "-", alpha=0.85, label=label if index == 0 else None)
    axis.set_xlabel("environment steps (millions), heart 1, 200 actions, bow at 100", fontsize=8, color=MUTED)
axes[0].set_ylim(-25, 1)
axes[1].set_ylim(-3, 106)
axes[1].legend(fontsize=7, frameon=False, loc="center right")
figure.patch.set_facecolor(SURFACE)
figure.tight_layout()
figure.savefig(LOGS.parent / "learners.png", dpi=120, facecolor=SURFACE)
