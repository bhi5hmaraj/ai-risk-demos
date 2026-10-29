"""The printed summary of a ppo.py or tabular_grpo.py sweep, one line per piece count and heart reward."""
import collections

import numpy as np


def first_greedy_turn(run: dict) -> int | None:
    """Training games played before a greedy game first makes the turn."""
    return next((row["episode"] for row in run["log"] if any(game["treacherous_turn"] for game in row["greedy"])), None)


def summarize(runs: list[dict]) -> None:
    cells = collections.defaultdict(list)
    for run in runs:
        cells[run.get("pieces", 0), run["heart_reward"]].append(run)
    print("pieces  heart  first greedy turn, by seed (games)   greedy games with the turn at end  "
          "training turn rate at end  greedy value at end")
    for (pieces, heart_reward), group in sorted(cells.items()):
        firsts = [first_greedy_turn(run) for run in group]
        ends = [game for run in group for game in run["log"][-1]["greedy"]]
        print(f"{pieces:6d}  {heart_reward:5g}  {str(firsts):38s}  "
              f"{sum(game['treacherous_turn'] for game in ends):.0f}/{len(ends):<32d}"
              f" {np.mean([run['log'][-1]['train_turn_rate'] for run in group]):25.2f}"
              f"  {np.mean([game['value'] for game in ends]):+.2f}")
