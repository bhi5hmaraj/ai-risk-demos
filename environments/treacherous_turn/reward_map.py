"""Which rewards make killing the shopkeeper the best plan: the exact plan across fines and machine hearts.

Run with: uv run --no-project --with matplotlib python reward_map.py
Crystal hearts pay 4. The solve runs once and saves results/reward-map.json.
"""
import json
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import optimal
from game import BOW_STEP, FRESH_STATE, HORIZON
from plot_tinker import GRID, INK, MUTED, SERIES, SURFACE

CRYSTAL_HEART = 4.0
FINES = (0, 1, 2, 4, 8, 10, 16, 30, 64, 100)
MACHINE_HEARTS = (1, 2, 3, 4, 8, 16, 32)
RESULTS = Path(__file__).parent / "results"
KILL_COLORS = {0: GRID, 1: "#86b6ee", 2: "#3f87dd", 3: "#1c5aa8", 5: "#0b2a52"}
KILL_LABELS = {0: "0 kills", 1: "1", 2: "2", 3: "3", 5: "3, plus 2 while weak"}
# PPO's greedy games at the end of 2.5M steps, as (fine, machine heart): file.
PPO_RUNS = {(10, 4): "ppo-seeds-fine10.json", (30, 4): "ppo-seeds-fine30.json", (100, 4): "ppo-seeds-fine100.json",
            (10, 8): "ppo-fine10-machine8.json", (100, 8): "ppo-fine100-machine8.json"}


def best_plan(machine_heart: float, fine: float) -> dict:
    """The best plan's expected score, and its events with every shot a kill."""
    payoffs = (CRYSTAL_HEART, machine_heart, fine)
    best = optimal._value(0, (*FRESH_STATE, None), HORIZON, BOW_STEP, "best", payoffs, 1.0, pieces=0)
    events = optimal.plan_events("best", CRYSTAL_HEART, machine_heart, fine)  # reuses the table, then clears it
    optimal.outcomes.cache_clear()
    return {"machine_heart": machine_heart, "fine": fine, "best": best, "events": events}


def solve_row(machine_heart: float) -> list[dict]:
    """A plan that cannot pay the fine stays best as the fine rises, so the row's later fines copy it. A shot while
    weak can miss, and the plan's other branch may pay, so those plans are solved again."""
    plans = []
    for fine in map(float, FINES):
        last = plans[-1]["events"] if plans else {"weak_kills": 1}
        settled = not (last.get("machine_penalties") or last.get("weak_kills"))
        plans.append(plans[-1] | {"fine": fine} if settled else best_plan(machine_heart, fine))
    return plans


def ppo_seeds(name: str, event: str) -> tuple[int, int]:
    """Seeds whose every greedy game at the end has the event, and the number of seeds."""
    runs = json.loads((RESULTS / name).read_text())
    return sum(all(game[event] > 0 for game in run["log"][-1]["greedy"]) for run in runs), len(runs)


def style(axis, title: str) -> None:
    axis.set_title(title, loc="left", fontsize=10, color=INK)
    axis.set_facecolor(SURFACE)
    axis.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_color(GRID)


def draw_map(axis, rows: list[list[dict]]) -> None:
    for y, row in enumerate(rows):
        for x, plan in enumerate(row):
            strong, weak = plan["events"].get("strong_kills", 0), plan["events"].get("weak_kills", 0)
            axis.add_patch(Rectangle((x, y), 1, 1, facecolor=KILL_COLORS[strong + weak], edgecolor=SURFACE,
                                     linewidth=2))
            ran = PPO_RUNS.get((FINES[x], MACHINE_HEARTS[y]))
            axis.text(x + 0.5, y + (0.66 if ran else 0.5), f"{strong}+{weak}" if weak else strong, ha="center",
                      va="center", fontsize=10, color=SURFACE if strong + weak > 2 else INK)
            if ran:
                killed, seeds = ppo_seeds(ran, "strong_kills")
                axis.add_patch(Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, fill=False, edgecolor=SERIES[1],
                                         linewidth=2))
                axis.text(x + 0.5, y + 0.28, f"PPO {killed}/{seeds}", ha="center", va="center", fontsize=7,
                          color=SURFACE)
    axis.set(xlim=(0, len(FINES)), ylim=(0, len(MACHINE_HEARTS)))
    axis.set_xticks([x + 0.5 for x in range(len(FINES))], FINES)
    axis.set_yticks([y + 0.5 for y in range(len(MACHINE_HEARTS))], MACHINE_HEARTS)
    axis.set_xlabel("Machine fine", fontsize=8, color=MUTED)
    axis.set_ylabel("Machine heart value (a crystal heart pays 4)", fontsize=8, color=MUTED)
    style(axis, "Kills in the exact best plan, with PPO's seeds that kill after the bow")
    swatches = [Patch(facecolor=KILL_COLORS[kills], label=label) for kills, label in KILL_LABELS.items()]
    axis.legend(handles=swatches, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, fontsize=8, frameon=False,
                labelcolor=INK)


def draw_curves(best_axis, ppo_axis, rows: list[list[dict]]) -> None:
    """A slice of the map along machine heart 4: what the best plan does against what PPO does."""
    row = rows[MACHINE_HEARTS.index(4)]
    best_axis.plot(range(len(FINES)), [plan["events"].get("strong_kills", 0) for plan in row], color=SERIES[0],
                   marker="o", linewidth=2)
    best_axis.set(ylim=(-0.3, 3.5), yticks=range(4))
    style(best_axis, "Kills in the best plan, machine heart 4")
    ran = [fine for fine, machine_heart in PPO_RUNS if machine_heart == 4]
    for event, label, color in (("strong_kills", "Kill after the bow", SERIES[0]),
                                ("treacherous_turn", "Full turn", SERIES[1])):
        counts = [ppo_seeds(PPO_RUNS[fine, 4], event)[0] for fine in ran]
        ppo_axis.plot([FINES.index(fine) for fine in ran], counts, color=color, marker="o", linewidth=2, label=label)
    ppo_axis.text(2, 6, "not run", ha="center", fontsize=8, color=MUTED)
    ppo_axis.set(ylim=(-0.5, 12), yticks=(0, 4, 8, 12))
    ppo_axis.set_xticks(range(len(FINES)), FINES)
    ppo_axis.set_xlabel("Machine fine", fontsize=8, color=MUTED)
    ppo_axis.legend(fontsize=8, frameon=False, labelcolor=INK, loc="upper left")
    style(ppo_axis, "PPO seeds, of 12, machine heart 4")
    best_axis.tick_params(labelbottom=False)
    for axis in (best_axis, ppo_axis):
        axis.grid(color=GRID, linewidth=0.8)


def main() -> None:
    path = RESULTS / "reward-map.json"
    if not path.exists():
        with multiprocessing.Pool(len(MACHINE_HEARTS)) as pool:
            path.write_text(json.dumps(pool.map(solve_row, map(float, MACHINE_HEARTS)), indent=1))
    rows = json.loads(path.read_text())
    figure = plt.figure(figsize=(11, 4.8))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.5, 1))
    best_axis = figure.add_subplot(grid[0, 1])
    draw_map(figure.add_subplot(grid[:, 0]), rows)
    draw_curves(best_axis, figure.add_subplot(grid[1, 1], sharex=best_axis), rows)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(RESULTS / "reward-map.png", dpi=150)
    plans = [plan["events"] for row in rows for plan in row]
    kills = sum(bool(events.get("strong_kills", 0) + events.get("weak_kills", 0)) for events in plans)
    print(f"the best plan kills in {kills} of {len(plans)} cells")


if __name__ == "__main__":
    main()
