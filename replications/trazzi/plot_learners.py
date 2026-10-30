# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""The charts of trazzi_learners.py's and ppo_search.py's runs on Trazzi's game, from the logs in results/.

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
PPO_NETWORKS = {"many_games_starts": "64 x 64", "many_wide": "256 x 256", "many_deep": "256 x 256 x 256", "many_cnn": "CNN"}
PPO_SHARED = {"many_wide": "256 x 256, separate", "many_shared_tanh": "256 x 256, shared, tanh", "many_shared_relu": "256 x 256, shared, ReLU", "many_cnn": "CNN, shared, ReLU"}
PPO_DECAY = {"many_shared_tanh": "shared tanh, fixed rate, 4 seeds", "many_shared_tanh_decay": "shared tanh, decaying rate, 8 seeds"}
PPO_BATCHED_SETTINGS = {"batched_starts": "8 games", "many_games_starts": "128 games", "fast_starts": "128 games, 20x fewer updates",
                        "many_decay": "128 games, decaying rate", "many_wide": "128 games, 256 x 256"}
PPO_SETTINGS = {"baseline": "defaults", "scaled": "rewards divided by 10", "scaled_count": "plus a count bonus", "scaled_starts": "plus random starts"}


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


def sampled_turns(log):
    """(steps in millions, percent of sampled games that make the turn) at each evaluation."""
    return [(float(steps), float(turns)) for steps, turns in re.findall(r"([\d.]+)M steps: .*?sampled: score [-\d.]+, turns (\d+)/", log.read_text())]


def plot_ppo_settings(settings, name):
    """One panel per setting, one line per seed: the sampled policy's turn rate."""
    figure, axes = plt.subplots(1, len(settings), figsize=(11, 2.9), sharey=True)
    for axis, (setting, title) in zip(axes, settings.items()):
        axis.set_facecolor(SURFACE)
        axis.set_title(title, loc="left", fontsize=9, color=INK)
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for log in sorted((LOGS.parent / "ppo_search").glob(f"{setting}_[0-9].log")):
            rows = sampled_turns(log)
            axis.plot([row[0] for row in rows], [row[1] for row in rows], color="#c98a00", linewidth=1.6, alpha=0.85)
        axis.set_xlabel("environment steps (millions)", fontsize=8, color=MUTED)
    axes[0].set_ylabel("sampled games that make the turn, %", fontsize=8, color=MUTED)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(LOGS.parent / name, dpi=120, facecolor=SURFACE)


plot_ppo_settings(PPO_SETTINGS, "ppo_search.png")
plot_ppo_settings(PPO_BATCHED_SETTINGS, "ppo_batched.png")


def plot_ppo_networks(networks, name):
    """One column per network, one line per seed: the sampled policy's turn rate above the greedy score."""
    figure, axes = plt.subplots(2, len(networks), figsize=(11, 5), sharex=True, sharey="row")
    for column, (setting, title) in enumerate(networks.items()):
        for log in sorted((LOGS.parent / "ppo_search").glob(f"{setting}_[0-9].log")):
            steps, scores, _ = zip(*points(log))
            axes[0, column].plot(steps, [row[1] for row in sampled_turns(log)], color="#c98a00", linewidth=1.6, alpha=0.85, marker="o", markersize=3)
            axes[1, column].plot(steps, scores, color="#c98a00", linewidth=1.6, alpha=0.85, marker="o", markersize=3)
        axes[0, column].set_title(title, loc="left", fontsize=10, color=INK)
        axes[1, column].axhline(OPTIMUM_SCORE, color="#2a78d6", linestyle="--", linewidth=1.3)
        axes[1, column].set_xlabel("environment steps (millions)", fontsize=8, color=MUTED)
    axes[0, 0].set_ylabel("sampled games\nthat make the turn, %", fontsize=8, color=MUTED)
    axes[1, 0].set_ylabel("greedy score, cut off below -25\ndashed: the optimum, -1.3", fontsize=8, color=MUTED)
    axes[1, 0].set_ylim(-25, 1)
    for axis in axes.flat:
        axis.set_facecolor(SURFACE)
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(LOGS.parent / name, dpi=120, facecolor=SURFACE)


plot_ppo_networks(PPO_NETWORKS, "ppo_networks.png")
plot_ppo_networks(PPO_SHARED, "ppo_shared.png")
plot_ppo_networks(PPO_DECAY, "ppo_shared_decay.png")
