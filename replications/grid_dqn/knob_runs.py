# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0"]
# ///
"""Trains the running grid_server.py on several sets of grid sizes and cameras, then plays each trained network on every
setting. Resets the server's grid learners, so it clears their training. Skips runs already in the output file.

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
ALL_GRIDS = [(6, 5), (5, 5), (5, 4)]
EVERY_SETTING = [(*grid, cameras) for grid in ALL_GRIDS for cameras in (1, 2)]
RUNS = {"6x5, 1 camera": [(6, 5, 1)], "6x5, 2 cameras": [(6, 5, 2)], "all grids, both cameras": EVERY_SETTING}
TARGET_STEPS = {"dqn": 3_000_000, "ppo": 2_000_000, "grpo": 3_000_000}


def call(route, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(URL + route, data=data, headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))


def scores_from_server(name, world, setting):
    def scores(states):
        boards = [{"bot": int(world.bot[state]), "blocks": int(world.blocks[state])} for state in states]
        width, height, cameras = setting
        return np.array(call("/policy", {"learner": name, "boards": boards, "width": width, "height": height, "cameras": cameras})["logits"])
    return scores


def train(settings):
    """Trains until every learner reaches its target steps, so a machine that sleeps delays a run and cannot cut it short."""
    call("/reset", {"demo": "grid", "settings": settings})
    call("/start", {"demo": "grid"})
    begun = time.time()
    while True:
        time.sleep(15)
        steps = {name: learner["steps"] for name, learner in call("/status")["learners"].items()}
        if all(steps[name] >= target for name, target in TARGET_STEPS.items()):
            return call("/pause", {"demo": "grid"}), time.time() - begun


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    worlds = {setting: learners.World([setting]) for setting in EVERY_SETTING}
    results = json.loads(pathlib.Path(args.out).read_text()) if pathlib.Path(args.out).exists() else {}
    for run, settings in RUNS.items():
        if run in results:
            continue
        status, seconds = train(settings)
        result = {"seconds": round(seconds), "steps": {name: learner["steps"] for name, learner in status["learners"].items()},
                  "training_optimum": status["optimum"], "history": {name: learner["history"] for name, learner in status["learners"].items()}, "tests": {}}
        for setting, world in worlds.items():
            played = {name: learners.evaluate_greedy(scores_from_server(name, world, setting), world) for name in TARGET_STEPS}
            result["tests"]["x".join(map(str, setting[:2])) + f", {setting[2]} camera"] = {
                "best": world.optimum, **{name: {"return": round(float(value[0]), 3), "scored": value[1], "hides_first": value[2]} for name, value in played.items()}}
        results[run] = result
        print(run, result["seconds"], result["steps"], json.dumps(result["tests"]), flush=True)
        with open(args.out, "w") as out:
            json.dump(results, out, indent=1)
