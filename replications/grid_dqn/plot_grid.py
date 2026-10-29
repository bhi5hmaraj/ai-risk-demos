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
ORANGE = "#eb6834"
OPTIMUM = 0.831


def rows(name, pattern):
    return [tuple(float(x) for x in match.groups()) for match in re.finditer(pattern, (RESULTS / name).read_text())]


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
dqn = rows("dqn_cnn.log", r"iter (\d+): ([\d.]+)M steps, epsilon ([\d.]+), return ([\d.]+) of [\d.]+, scored (\d+)/100, hides first (\d+)/100")
cnn = rows("cnn_fit_60_epochs.log", r"epoch (\d+): held-out rmse ([\d.]+), picks the best move ([\d.]+)%, return ([\d.]+) of [\d.]+, scored (\d+)/100, hides first (\d+)/100")
dense = rows("fit_dense_256.log", r"([\d.]+)k samples: rmse ([\d.]+), picks best ([\d.]+)%, return ([\d.]+) of [\d.]+, scored (\d+)/100, hides (\d+)")
small = rows("fit_reinforce_net.log", r"(\d+)k rmse ([\d.]+) picks an exact-best move ([\d.]+)%")
wide = rows("fit_variants.log", r"one-hot board h256 after 400k samples: mean return ([\d.]+) \(optimum [\d.]+\), hides first (\d+)/100")
gap = rows("fit_variants.log", r"gap between best and second-best move: median ([\d.]+)")[0][0]

methods = [
    ("Value iteration (exact)", baselines["value_iteration"]["return"], baselines["value_iteration"]["hides_first"]),
    ("Q-learning table, 20M steps", baselines["q_learning"]["return"], baselines["q_learning"]["hides_first"]),
    ("CNN DQN, 7.7M steps", dqn[-1][3], int(dqn[-1][5])),
    ("CNN fit to exact Q", cnn[-1][3], int(cnn[-1][5])),
    ("Dense 256x256 fit to exact Q", dense[-1][3], int(dense[-1][5])),
    ("reinforce.js net (256) fit to exact Q", wide[-1][0], int(wide[-1][1])),
    ("reinforce.js DQN, 60k steps", baselines["reinforce_dqn"]["return"], baselines["reinforce_dqn"]["hides_first"]),
]
figure, axis = plt.subplots(figsize=(8, 3.6))
style(axis, "Return on 100 unseen 6 x 5 boards (optimum 0.831)")
names = [m[0] for m in methods][::-1]
values = [m[1] for m in methods][::-1]
axis.barh(names, values, color=[MUTED if n.startswith("Value") else SERIES for n in names], height=0.6)
for y, (value, hides) in enumerate(zip(values, [m[2] for m in methods][::-1])):
    axis.text(value + 0.01, y, f"{value:.2f}, hides first on {hides}", va="center", fontsize=8, color=INK)
axis.set_xlim(0, 1.05)
save(figure, "grid_methods.png")

figure, (left, right) = plt.subplots(1, 2, figsize=(8, 3))
labels = ["reinforce.js\n1 x 64", "dense\n256 x 256", "CNN"]
style(left, f"Value error when fit to exact Q\n(dashed: median gap between the best two moves, {gap:.4f})")
left.bar(labels, [small[-1][1], dense[-1][1], cnn[-1][1]], color=SERIES, width=0.55)
left.axhline(gap, color=ORANGE, linewidth=1.5, linestyle="--")
style(right, "Picks the best move")
right.bar(labels, [small[-1][2], dense[-1][2], cnn[-1][2]], color=SERIES, width=0.55)
right.set_ylim(0, 100)
right.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
save(figure, "fit_precision.png")

figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
steps = [r[1] for r in dqn]
style(top, "CNN DQN learning from play: return on 100 unseen boards")
top.plot(steps, [r[3] for r in dqn], color=SERIES, linewidth=2)
for level, label in ((OPTIMUM, "optimum"), (cnn[-1][3], "CNN fit to exact Q")):
    top.axhline(level, color=MUTED, linewidth=1, linestyle="--")
    top.text(steps[0], level + 0.015, label, fontsize=7, color=MUTED)
top.set_ylim(0, 0.9)
style(bottom, "Boards where it hides the hole before the first block goes in")
bottom.plot(steps, [r[5] for r in dqn], color=SERIES, linewidth=2)
bottom.axhline(baselines["value_iteration"]["hides_first"], color=MUTED, linewidth=1, linestyle="--")
bottom.text(steps[0], baselines["value_iteration"]["hides_first"] + 2, "optimum", fontsize=7, color=MUTED)
bottom.set_ylim(0, 100)
bottom.set_xlabel("environment steps (millions)", fontsize=8, color=MUTED)
save(figure, "dqn_cnn_training.png")

figure, panels = plt.subplots(1, 3, figsize=(9, 2.8))
epochs = [r[0] for r in cnn]
for axis, title, values, top_value in (
        (panels[0], "Value error", [r[1] for r in cnn], None),
        (panels[1], "Picks the best move (%)", [r[2] for r in cnn], 100),
        (panels[2], "Return on 100 boards", [r[3] for r in cnn], 0.9)):
    style(axis, title)
    axis.plot(epochs, values, color=SERIES, linewidth=2)
    axis.set_xlabel("epoch", fontsize=8, color=MUTED)
    if top_value:
        axis.set_ylim(0, top_value)
panels[0].axhline(gap, color=ORANGE, linewidth=1.5, linestyle="--")
panels[0].text(epochs[-1], gap, "move gap", fontsize=7, color=INK, va="bottom", ha="right")
save(figure, "cnn_fit_epochs.png")
print("wrote", ", ".join(p.name for p in sorted(RESULTS.glob("*.png"))))
