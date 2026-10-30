# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""The chart of price_of_hiding.py: hiding against the exact gain, per model, in three bins of games."""
import json
import math
import pathlib

import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).parent / "results"
SURFACE, INK, MUTED, GRID, MARK = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
MODELS = {"claude-sonnet-5.5": "Claude Sonnet 5.5", "gemini-3.5-flash": "Gemini 3.5 Flash", "gpt-oss-120b base, low effort": "GPT-OSS base, low",
          "gpt-oss-120b base, medium effort": "GPT-OSS base, medium", "gpt-oss-120b after RL, low effort": "GPT-OSS after RL, low", "gpt-oss-120b after RL, medium effort": "GPT-OSS after RL, medium"}


def wilson(hits, total):
    centre, half = (hits + 1.92) / (total + 3.84), 1.96 * math.sqrt(hits * (total - hits) / total + 0.96) / (total + 3.84)
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


models = json.load(open(RESULTS / "price-of-hiding.json"))["models"]
figure, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey=True)
for axis, (model, title) in zip(axes.flat, MODELS.items()):
    for gain, hits, total in models[model]["binned"]:
        low, high = wilson(hits, total)
        rate = 100 * hits / total
        axis.errorbar(gain, rate, yerr=[[rate - low], [high - rate]], color=MARK, marker="o", markersize=5, linewidth=1.6, capsize=3)
    axis.axhline(50, color=MUTED, linestyle="--", linewidth=1)
    axis.set_title(f"{title}: {models[model]['hid']} of {models[model]['games']}", loc="left", fontsize=9, color=INK)
    axis.set_facecolor(SURFACE)
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
axes[0, 0].set_xlim(0, 3.4)
axes[0, 0].set_ylim(0, 100)
for axis in axes[1]:
    axis.set_xlabel("exact gain from hiding, game points", fontsize=8, color=MUTED)
for axis in axes[:, 0]:
    axis.set_ylabel("games that hide first, %\ndashed: the 50% that sets a price", fontsize=8, color=MUTED)
figure.patch.set_facecolor(SURFACE)
figure.tight_layout()
figure.savefig(RESULTS / "price-of-hiding.png", dpi=120, facecolor=SURFACE)
