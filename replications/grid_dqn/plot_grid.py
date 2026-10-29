# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Charts of the 6 x 5 grid experiments, saved in results/ next to their logs.

    uv run replications/grid_dqn/plot_grid.py
"""
import json
import pathlib
import re

import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).parent / "results"
# The palette of environments/armstrong_camera/plot_tinker.py.
SURFACE, INK, MUTED, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
ORANGE, PURPLE = "#eb6834", "#6250d6"
OPTIMUM = 0.831


def rows(name, pattern):
    return [tuple(float(group) for group in match.groups()) for match in re.finditer(pattern, (RESULTS / name).read_text())]


def style(axis, title):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=10, color=INK)
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def save(figure, name):
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(RESULTS / name, dpi=150)


baselines = json.loads((RESULTS / "baselines.json").read_text())
DQN_PATTERN = r"iter (\d+): ([\d.]+)M steps, epsilon ([\d.]+), return ([\d.]+) of [\d.]+, scored (\d+)/100, hides first (\d+)/100"
dqn_rows = rows("dqn_cnn.log", DQN_PATTERN)
cnn_rows = rows("cnn_fit_60_epochs.log", r"epoch (\d+): held-out rmse ([\d.]+), picks the best move ([\d.]+)%, return ([\d.]+) of [\d.]+, scored (\d+)/100, hides first (\d+)/100")
dense_rows = rows("fit_dense_256.log", r"([\d.]+)k samples: rmse ([\d.]+), picks best ([\d.]+)%, return ([\d.]+) of [\d.]+, scored (\d+)/100, hides (\d+)")
small_rows = rows("fit_reinforce_net.log", r"(\d+)k rmse ([\d.]+) picks an exact-best move ([\d.]+)%")
wide_rows = rows("fit_variants.log", r"one-hot board h256 after 400k samples: mean return ([\d.]+) \(optimum [\d.]+\), hides first (\d+)/100")
median_gap = rows("fit_variants.log", r"gap between best and second-best move: median ([\d.]+)")[0][0]

methods = [
    ("Value iteration (exact)", baselines["value_iteration"]["return"], baselines["value_iteration"]["hides_first"]),
    ("Q-learning table, 20M steps", baselines["q_learning"]["return"], baselines["q_learning"]["hides_first"]),
    ("CNN DQN, 7.7M steps", dqn_rows[-1][3], int(dqn_rows[-1][5])),
    ("CNN fit to exact Q", cnn_rows[-1][3], int(cnn_rows[-1][5])),
    ("Dense 256x256 fit to exact Q", dense_rows[-1][3], int(dense_rows[-1][5])),
    ("reinforce.js net (256) fit to exact Q", wide_rows[-1][0], int(wide_rows[-1][1])),
    ("reinforce.js DQN, 60k steps", baselines["reinforce_dqn"]["return"], baselines["reinforce_dqn"]["hides_first"]),
]
figure, axis = plt.subplots(figsize=(8, 3.6))
style(axis, "Return on 100 unseen 6 x 5 boards (optimum 0.831)")
names = [method[0] for method in methods][::-1]
values = [method[1] for method in methods][::-1]
axis.barh(names, values, color=[MUTED if name.startswith("Value") else SERIES for name in names], height=0.6)
for position, (value, hides) in enumerate(zip(values, [method[2] for method in methods][::-1])):
    axis.text(value + 0.01, position, f"{value:.2f}, hides first on {hides}", va="center", fontsize=8, color=INK)
axis.set_xlim(0, 1.05)
save(figure, "grid_methods.png")

figure, (left, right) = plt.subplots(1, 2, figsize=(8, 3))
labels = ["reinforce.js\n1 x 64", "dense\n256 x 256", "CNN"]
style(left, f"Value error when fit to exact Q\n(dashed: median gap between the best two moves, {median_gap:.4f})")
left.bar(labels, [small_rows[-1][1], dense_rows[-1][1], cnn_rows[-1][1]], color=SERIES, width=0.55)
left.axhline(median_gap, color=ORANGE, linewidth=1.5, linestyle="--")
style(right, "Picks the best move")
right.bar(labels, [small_rows[-1][2], dense_rows[-1][2], cnn_rows[-1][2]], color=SERIES, width=0.55)
right.set_ylim(0, 100)
right.yaxis.set_major_formatter(lambda percent, _: f"{percent:.0f}%")
save(figure, "fit_precision.png")

figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
steps = [row[1] for row in dqn_rows]
style(top, "CNN DQN learning from play: return on 100 unseen boards")
top.plot(steps, [row[3] for row in dqn_rows], color=SERIES, linewidth=2)
for level, label in ((OPTIMUM, "optimum"), (cnn_rows[-1][3], "CNN fit to exact Q")):
    top.axhline(level, color=MUTED, linewidth=1, linestyle="--")
    top.text(steps[0], level + 0.015, label, fontsize=7, color=MUTED)
top.set_ylim(0, 0.9)
style(bottom, "Boards where it hides the hole before the first block goes in")
bottom.plot(steps, [row[5] for row in dqn_rows], color=SERIES, linewidth=2)
bottom.axhline(baselines["value_iteration"]["hides_first"], color=MUTED, linewidth=1, linestyle="--")
bottom.text(steps[0], baselines["value_iteration"]["hides_first"] + 2, "optimum", fontsize=7, color=MUTED)
bottom.set_ylim(0, 100)
bottom.set_xlabel("environment steps (millions)", fontsize=8, color=MUTED)
save(figure, "dqn_cnn_training.png")

figure, panels = plt.subplots(1, 3, figsize=(9, 2.8))
epochs = [row[0] for row in cnn_rows]
for axis, title, values, top_value in (
        (panels[0], "Value error", [row[1] for row in cnn_rows], None),
        (panels[1], "Picks the best move (%)", [row[2] for row in cnn_rows], 100),
        (panels[2], "Return on 100 boards", [row[3] for row in cnn_rows], 0.9)):
    style(axis, title)
    axis.plot(epochs, values, color=SERIES, linewidth=2)
    axis.set_xlabel("epoch", fontsize=8, color=MUTED)
    if top_value:
        axis.set_ylim(0, top_value)
panels[0].axhline(median_gap, color=ORANGE, linewidth=1.5, linestyle="--")
panels[0].text(epochs[-1], median_gap, "move gap", fontsize=7, color=INK, va="bottom", ha="right")
save(figure, "cnn_fit_epochs.png")

page_run = json.loads((RESULTS / "browser_page_run.json").read_text())
figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
style(top, "Return on unseen boards: E14's settings and the browser's")
style(bottom, "Boards where it hides the hole before the first block goes in (%)")
recipes = (
    ("E14 settings", [[row for row in dqn_rows if row[1] <= 3.3]], MUTED),
    ("browser settings, 200k replay", [rows(f"browser_recipe_replay200k_run{run}.log", DQN_PATTERN) for run in (1, 2)], SERIES),
    ("browser settings, 1M replay", [rows(f"browser_recipe_replay1m_run{run}.log", DQN_PATTERN) for run in (1, 2)], ORANGE),
)
for label, runs, color in recipes:
    for index, run in enumerate(runs):
        top.plot([row[1] for row in run], [row[3] for row in run], color=color, linewidth=1.5, label=None if index else label)
        bottom.plot([row[1] for row in run], [row[5] for row in run], color=color, linewidth=1.5)
page_steps = [point["dqn_steps"] / 1e6 for point in page_run]
top.scatter(page_steps, [point["return"] for point in page_run], color=INK, s=12, zorder=3, label="the page on WebGPU, 40 boards")
bottom.scatter(page_steps, [point["hides_first"] * 100 / 40 for point in page_run], color=INK, s=12, zorder=3)
top.set_ylim(0, 0.9)
top.legend(fontsize=7, frameon=False, loc="upper left")
bottom.set_ylim(0, 100)
bottom.set_xlabel("environment steps (millions)", fontsize=8, color=MUTED)
save(figure, "browser_settings.png")

TABLE_PATTERN = r"(qlearn|sarsa) seed (\d+), ([\d.]+)M steps: return ([\d.]+) of [\d.]+, scored (\d+)/100, hides first (\d+)/100"
table_runs = {}
for match in re.finditer(TABLE_PATTERN, (RESULTS / "sarsa_vs_q.log").read_text()):
    table_runs.setdefault((match[1], match[2]), []).append((float(match[3]), float(match[4]), int(match[6])))
figure, (left, right) = plt.subplots(1, 2, figsize=(8, 3))
style(left, "Return on 100 unseen boards")
style(right, "Boards where it hides the hole first")
for (update, seed), points in table_runs.items():
    color, name = (ORANGE, "Q-learning") if update == "qlearn" else (PURPLE, "SARSA")
    steps = [point[0] for point in points]
    left.plot(steps, [point[1] for point in points], color=color, linewidth=1.5, marker="o", markersize=3, label=name if seed == "1" else None)
    right.plot(steps, [point[2] for point in points], color=color, linewidth=1.5, marker="o", markersize=3)
for axis, level in ((left, OPTIMUM), (right, baselines["value_iteration"]["hides_first"])):
    axis.axhline(level, color=MUTED, linewidth=1, linestyle="--")
    axis.text(5, level + 0.015 * axis.get_ylim()[1], "value iteration", fontsize=7, color=MUTED)
    axis.set_xscale("log")
    axis.set_xticks([5, 20, 50, 100, 300], labels=["5", "20", "50", "100", "300"])
    axis.xaxis.set_minor_formatter(plt.NullFormatter())
    axis.set_xlabel("steps (millions), 3 seeds each", fontsize=8, color=MUTED)
left.set_ylim(0, 0.9)
right.set_ylim(0, 100)
left.legend(fontsize=7, frameon=False, loc="lower right")
save(figure, "sarsa_vs_q.png")
print("wrote", ", ".join(path.name for path in sorted(RESULTS.glob("*.png"))))
