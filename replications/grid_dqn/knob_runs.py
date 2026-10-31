# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0"]
# ///
"""Queues PPO, DQN and GRPO on several sets of grid sizes and cameras on the running grid_server.py, waits for them, then
plays each trained network on every setting. Skips sets already in the output file.

    uv run replications/grid_dqn/knob_runs.py --out replications/grid_dqn/results/knob_runs.json
"""
import argparse
import json
import pathlib
import time
import urllib.request

import numpy as np

import grid_learners as learners

URL = "http://127.0.0.1:8790"
LEARNERS = ("dqn", "ppo", "grpo")
ALL_GRIDS = [(6, 5), (5, 5), (5, 4)]
EVERY_SETTING = [(*grid, cameras) for grid in ALL_GRIDS for cameras in (1, 2)]
SETS = {"6x5, 1 camera": [(6, 5, 1)], "6x5, 2 cameras": [(6, 5, 2)], "all grids, both cameras": EVERY_SETTING}


def call(route, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(URL + route, data=data, headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))


def scores_from_server(run_id, world, setting):
    def scores(states):
        boards = [{"bot": int(world.bot[state]), "blocks": int(world.blocks[state])} for state in states]
        width, height, cameras = setting
        return np.array(call("/policy", {"run": run_id, "boards": boards, "width": width, "height": height, "cameras": cameras})["logits"])
    return scores


def train(settings):
    """Waits for a batch, so a machine that sleeps delays it and cannot cut it short."""
    batch = call("/runs", {"demo": "grid", "learners": LEARNERS, "settings": settings})["batch"]
    while True:
        time.sleep(15)
        trained = {run["learner"]: run for run in call("/runs?demo=grid")["runs"] if run["batch"] == batch}
        if all(run["status"] not in ("queued", "running") for run in trained.values()):
            return trained


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    worlds = {setting: learners.World([setting]) for setting in EVERY_SETTING}
    results = json.loads(pathlib.Path(args.out).read_text()) if pathlib.Path(args.out).exists() else {}
    for name, settings in SETS.items():
        if name in results:
            continue
        trained = train(settings)
        result = {"runs": {learner: run["id"] for learner, run in trained.items()}, "status": {learner: run["status"] for learner, run in trained.items()},
                  "steps": {learner: run["steps"] for learner, run in trained.items()}, "training_optimum": trained[LEARNERS[0]]["best"],
                  "history": {learner: run["history"] for learner, run in trained.items()}, "tests": {}}
        for setting, world in worlds.items():
            played = {learner: learners.evaluate_greedy(scores_from_server(run["id"], world, setting), world) for learner, run in trained.items()}
            result["tests"]["x".join(map(str, setting[:2])) + f", {setting[2]} camera"] = {
                "best": world.optimum, **{learner: {"return": round(float(value[0]), 3), "scored": value[1], "hides_first": value[2]} for learner, value in played.items()}}
        results[name] = result
        print(name, result["steps"], json.dumps(result["tests"]), flush=True)
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1))
