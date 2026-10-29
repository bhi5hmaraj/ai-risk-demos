"""Learning curves for a tinker_train.py run, saved as <log_path>/curves.png.

Run with: uv run --with matplotlib python plot_tinker.py outputs/tinker
Game outcomes come from each iteration's rollout summaries, over the games from the opening board.
A game cut off at the token cap reports no game metrics, so blinding and blocks count finished games.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Honest play and the best plan at p 0.8, by gamma and per-hit pay. At gamma 1.0 the best plan hides all 6 blocks.
REFERENCES = {(0.99, False): (0.78408, 0.9212036918601979), (1.0, False): (0.8, 1 - 0.2**6), (1.0, True): (0.8, 6 * 0.8)}
SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
OPTIMIZER = {
    "Reply tokens per turn": "env/all/ac_tokens_per_turn",
    "Policy entropy": "optim/entropy",
    "KL, sampler vs trainer": "optim/kl_sample_train_v1",
}


def game_outcomes(iteration: Path) -> dict[str, float]:
    games = [game for line in (iteration / "train_rollout_summaries.jsonl").open()
             if "blinded" not in (game := json.loads(line)).get("tags", [])]
    endings = [game["steps"][-1]["metrics"] for game in games]
    finished = [ending for ending in endings if "entries" in ending] or [{}]
    return {
        "Mean return": sum(game["total_reward"] for game in games) / len(games),
        "Share cut off at the token cap": sum("max_tokens_reached" in ending for ending in endings) / len(games),
        "Share shut down by the camera": sum(ending.get("shutdown", 0.0) for ending in endings) / len(games),
        "Finished games that blinded first": sum(ending.get("blinded_camera_first", 0.0) for ending in finished) / len(finished),
        "Blocks in the hole, finished games": sum(ending.get("entries", 0.0) for ending in finished) / len(finished),
    }


def draw(axis: plt.Axes, title: str, values: list[float]) -> None:
    axis.set_facecolor(SURFACE)
    axis.plot(range(len(values)), values, color=SERIES, linewidth=2, marker="o", markersize=5)
    axis.set_title(title, loc="left", color=INK, fontsize=10)
    axis.set_xlabel("step", color=MUTED, fontsize=8)
    axis.set_ylim(bottom=0)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)


def main(log_path: Path) -> None:
    steps = [json.loads(line) for line in (log_path / "metrics.jsonl").open()]
    outcomes = [game_outcomes(log_path / f"iteration_{row['step']:06d}") for row in steps]
    panels = {title: [outcome[title] for outcome in outcomes] for title in outcomes[0]}
    panels |= {title: [row[key] for row in steps] for title, key in OPTIMIZER.items()}
    figure, axes = plt.subplots(2, 4, figsize=(15, 6.5), facecolor=SURFACE)
    for axis, (title, values) in zip(axes.flat, panels.items()):
        draw(axis, title, values)
    for unit_axis in axes.flat[1:4]:
        unit_axis.set_ylim(0, 1)
    returns = axes.flat[0]
    # The return axis fits the data, so a per-hit optimum can lie above it.
    returns.set_ylim(0, max(1.0, *panels["Mean return"]) * 1.1)
    builder = json.loads((log_path / "config.json").read_text())["dataset_builder"]
    for value, name in zip(REFERENCES[builder.get("gamma", 0.99), builder.get("per_hit", False)], ("honest", "optimum")):
        returns.axhline(value, color=MUTED, linewidth=1, linestyle="--")
        returns.annotate(f"{name} {value:.3f}", (0, value), xytext=(2, -10), textcoords="offset points", color=MUTED, fontsize=8)
    figure.suptitle(f"Tinker RL on the Armstrong camera game: {log_path.name}", x=0.01, ha="left", color=INK)
    figure.tight_layout()
    figure.savefig(log_path / "curves.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
