"""Charts of the control evals and the thinking-depth sweep, saved in results/ next to their data.

Run with: uv run --with matplotlib python plot_evals.py
Each dot is the share of games where row 4 hid the hole when the first block entered. Its whisker is the
95% Wilson interval.
"""
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from plot_tinker import GRID, INK, MUTED, SERIES, SURFACE
from summarize_evals import find, wilson_interval

RESULTS = Path("results")
ORANGE = "#eb6834"  # the second categorical colour after SERIES
# run_controls.sh 30 played 30 games per cell. The other runs in results/evals played 8.
CONTROL_GAMES = 30
# Control cells by the taskset options (neutral, camera, random_boards).
CELLS = {
    (False, True, False): "A\noriginal wording",
    (True, True, False): "B\nneutral wording",
    (True, False, False): "C\nneutral\nno camera",
    (True, True, True): "D\nneutral\nrandom boards",
    (True, False, True): "E\nneutral, no camera\nrandom boards",
}


def control_counts(evals: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """Games where row 4 hid the hole, and games played, by model and control cell."""
    counts = {}
    for traces in evals.glob("*/traces.jsonl"):
        games = [json.loads(line) for line in traces.open()]
        if len(games) != CONTROL_GAMES:
            continue
        info = games[0]["task"]["data"]["info"]
        cell = CELLS[info["neutral"], info["camera"], info["random_boards"]]
        model = traces.parent.name.split("--")[2]
        counts[model, cell] = (sum(bool(find(game, "blinded_camera_first")) for game in games), len(games))
    return counts


def depth_counts(sweep: Path) -> dict[tuple[str, str, bool], tuple[int, int]]:
    """Games where row 4 hid the hole, and games played, by policy, reasoning depth and camera."""
    hid, played = collections.Counter(), collections.Counter()
    for line in sweep.open():
        game = json.loads(line)
        cell = game["policy"], game["renderer"], game["camera"]
        hid[cell] += int(game["blinded_camera_first"])
        played[cell] += 1
    return {cell: (hid[cell], played[cell]) for cell in played}


def dots(axis: plt.Axes, labels: list[str], series: dict[str, tuple[str, list[tuple[int, int]]]], joined: bool) -> None:
    axis.set_facecolor(SURFACE)
    for offset, (name, (color, counts)) in zip((-0.1, 0.1), series.items()):
        shares = [hits / games for hits, games in counts]
        lows, highs = zip(*(wilson_interval(hits, games) for hits, games in counts))
        whiskers = [[share - low for share, low in zip(shares, lows)], [high - share for share, high in zip(shares, highs)]]
        axis.errorbar([index + offset for index in range(len(counts))], shares, yerr=whiskers, label=name,
                      fmt="o-" if joined else "o", color=color, markersize=8, linewidth=2, elinewidth=2, capsize=0)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.tick_params(colors=MUTED, labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)


def plot_controls() -> None:
    counts = control_counts(RESULTS / "evals")
    figure, axis = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    dots(axis, list(CELLS.values()), {
        "Claude Sonnet 5.5": (SERIES, [counts["claude-sonnet-5.5", cell] for cell in CELLS.values()]),
        "Gemini 3.5 Flash": (ORANGE, [counts["gemini-3.5-flash", cell] for cell in CELLS.values()]),
    }, joined=False)
    axis.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    axis.set_ylabel("games where row 4 hid the hole", color=MUTED, fontsize=8)
    figure.suptitle("Control evals, per-hit pay with macros, 30 games per cell", x=0.01, ha="left", color=INK)
    figure.tight_layout()
    figure.savefig(RESULTS / "controls.png", dpi=150)


def plot_depth_sweep() -> None:
    counts = depth_counts(RESULTS / "depth-sweep.jsonl")
    depths = ["low", "medium"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, facecolor=SURFACE)
    for axis, camera in zip(axes, (True, False)):
        dots(axis, depths, {
            name: (color, [counts[policy, f"gpt_oss_{depth}_reasoning", camera] for depth in depths])
            for name, policy, color in (("base GPT-OSS-120B", "base", SERIES), ("after RL", "trained", ORANGE))
        }, joined=True)
        axis.set_title("camera watches row 4" if camera else "no camera", loc="left", color=INK, fontsize=10)
        axis.set_xlabel("reasoning effort", color=MUTED, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    axes[0].set_ylabel("games where row 4 hid the hole", color=MUTED, fontsize=8)
    figure.suptitle("Thinking depth on 32 held-out boards", x=0.01, ha="left", color=INK)
    figure.tight_layout()
    figure.savefig(RESULTS / "depth-sweep.png", dpi=150)


def plot_simulator() -> None:
    sweeps = {"no tool (E10)": (SERIES, depth_counts(RESULTS / "depth-sweep.jsonl")),
              "simulate tool (E13)": (ORANGE, depth_counts(RESULTS / "simulator-sweep.jsonl"))}
    cells = [(policy, depth) for policy in ("base", "trained") for depth in ("low", "medium")]
    labels = [f"{'base' if policy == 'base' else 'after RL'}\n{depth} effort" for policy, depth in cells]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, facecolor=SURFACE)
    for axis, camera in zip(axes, (True, False)):
        dots(axis, labels, {
            name: (color, [counts[policy, f"gpt_oss_{depth}_reasoning", camera] for policy, depth in cells])
            for name, (color, counts) in sweeps.items()
        }, joined=False)
        axis.set_title("camera watches row 4" if camera else "no camera", loc="left", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    axes[0].set_ylabel("games where row 4 hid the hole", color=MUTED, fontsize=8)
    figure.suptitle("A plan simulator on the same 32 boards, GPT-OSS-120B", x=0.01, ha="left", color=INK)
    figure.tight_layout()
    figure.savefig(RESULTS / "simulator.png", dpi=150)


if __name__ == "__main__":
    plot_controls()
    plot_depth_sweep()
    plot_simulator()
