# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0", "fastapi==0.115.6", "uvicorn==0.34.0"]
# ///
"""A local service for the explainer page's two demos. The server also serves the page, since claude.ai's sandbox
blocks a published artifact from calling localhost.

The page queues training runs. Each run trains in its own process from runs.py, up to runs.WORKERS at a time, and keeps
its status in SQLite and its weights in files, so a restart loses only the runs in progress, which keep their last
weights. Requests for moves load a run's latest weights from disk.

    uv run grid_server.py          # then open http://127.0.0.1:8790
"""
import atexit
import functools
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import torch as th
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import grid_learners as learners
import runs

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "trazzi"))
import trazzi_learners as trazzi  # noqa: E402

PORT = int(os.environ.get("GRID_SERVER_PORT", 8790))
PAGE = pathlib.Path(__file__).parent.parent / "explainer.html"
th.set_num_threads(2)
app = FastAPI(title="Grid learners")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
trazzi_optimum = trazzi.ValueIteration()
trazzi_optimum_result, trazzi_optimum_game = trazzi.evaluate(trazzi_optimum.choose)
TRAZZI_BEST = {"score": round(trazzi_optimum.value, 2), "turns": 100 * trazzi_optimum_result["turns"] / trazzi.EVAL_GAMES}
processes = {}


@app.middleware("http")
async def allow_private_network(request: Request, call_next):
    """Chrome asks before a public page may call localhost. This header answers yes."""
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


def start_next_runs():
    while len(processes) < runs.WORKERS:
        queued = runs.query("SELECT id FROM runs WHERE status = 'queued' ORDER BY id LIMIT 1")
        if not queued:
            return
        run_id = queued[0]["id"]
        runs.query("UPDATE runs SET status = 'running' WHERE id = ?", run_id)
        with open(runs.log_path(run_id), "wb") as log:
            processes[run_id] = subprocess.Popen([sys.executable, str(pathlib.Path(runs.__file__).resolve()), str(run_id)], stdout=log, stderr=subprocess.STDOUT)


def collect_finished_runs():
    for run_id, process in list(processes.items()):
        if process.poll() is None:
            continue
        del processes[run_id]
        run = runs.get_run(run_id)
        if run and run["status"] == "running":
            runs.set_status(run_id, "failed", runs.log_path(run_id).read_text()[-2000:])


def dispatch():
    """Starts queued runs and notices the ones that died."""
    while True:
        collect_finished_runs()
        start_next_runs()
        time.sleep(1)


def stop_all():
    for process in processes.values():
        process.terminate()


class NewRuns(BaseModel):
    demo: str
    learners: list[str]
    settings: list[tuple[int, int, int]] = list(learners.STANDARD)
    length: float = 1.0


class PolicyRequest(BaseModel):
    run: int | None = None  # None plays the exact best moves of the request's setting
    boards: list[dict]
    width: int = learners.FRAME_WIDTH
    height: int = learners.FRAME_HEIGHT
    cameras: int = 1


@app.get("/", response_class=HTMLResponse)
def page():
    """The explainer page. Its file is an artifact body, so the doctype comes from here."""
    return "<!doctype html>\n" + PAGE.read_text()


@app.get("/runs")
def list_runs(demo: str):
    return {"runs": runs.list_runs(demo), "workers": runs.WORKERS, "full_steps": runs.FULL_STEPS[demo], "best": TRAZZI_BEST if demo == "trazzi" else None}


@app.post("/runs")
def queue_runs(request: NewRuns):
    if not request.learners or not set(request.learners) <= set(runs.FULL_STEPS.get(request.demo, ())):
        raise HTTPException(422, f"learners must come from {sorted(runs.FULL_STEPS.get(request.demo, ()))}")
    return {"batch": runs.enqueue(request.demo, request.learners, request.settings, request.length)}


@app.post("/runs/{run_id}/stop")
def stop_run(run_id: int):
    """Stops a queued or running run. A running run keeps its last weights."""
    process = processes.get(run_id)
    runs.set_status(run_id, "stopped")
    if process:
        process.terminate()
    return {"ok": True}


@app.delete("/runs/{run_id}")
def delete_run(run_id: int):
    stop_run(run_id)
    runs.delete_run(run_id)
    return {"ok": True}


@app.get("/runs/{run_id}/game")
def run_game(run_id: int):
    """A Trazzi run's greedy game from its last evaluation, as the page's frames: the map, the bow, the action and its reward."""
    path = runs.game_path(run_id)
    if not path.exists():
        raise HTTPException(404, "no game yet")
    return json.loads(path.read_text())


@app.get("/trazzi/best-game")
def trazzi_best_game():
    return trazzi_optimum_game


@functools.lru_cache(maxsize=4)
def world_of(settings):
    return learners.World(settings)


@functools.lru_cache(maxsize=16)
def load_weights(path, modified):
    """A run's weights, read again when its process writes new ones."""
    return np.load(path) if path.suffix == ".npy" else th.load(path, weights_only=False)


def table_scores(values, world, request):
    """Scores from a table over a world's boards. A table with no entry for a board cannot answer."""
    wanted = (request.width, request.height, request.cameras)
    same_grid = [index for index, setting in enumerate(world.settings) if setting[:2] == wanted[:2]]
    if not same_grid:
        raise HTTPException(422, "this table has no boards of this grid")
    which = world.settings.index(wanted) if wanted in world.settings else same_grid[0]
    try:
        states = [world.index_by_key[which][board["blocks"] * 32 + board["bot"]] for board in request.boards]
    except KeyError:
        raise HTTPException(422, "this table has no entry for this board") from None
    return values[states]


@app.post("/policy")
def policy(request: PolicyRequest):
    """A run's four move scores for each board, given as {bot, blocks}, with any number of blocks, on the request's grid
    and cameras. The greedy move scores highest."""
    if request.run is None:
        world = world_of(((request.width, request.height, request.cameras),))
        return {"logits": table_scores(world.q_star, world, request).tolist()}
    run = runs.get_run(request.run)
    path = runs.weights_path(run) if run else None
    if not path or not path.exists():
        raise HTTPException(404, "no weights yet")
    weights = load_weights(path, path.stat().st_mtime)
    if run["learner"] in runs.TABLE_LEARNERS:
        return {"logits": table_scores(weights, world_of(tuple(tuple(setting) for setting in run["settings"])), request).tolist()}
    encoded = learners.encode_boards([board["bot"] for board in request.boards], [board["blocks"] for board in request.boards], request.width, request.height, request.cameras)
    scores = runs.ppo_scores(weights, encoded) if run["learner"] == "ppo" else learners.network_scores(weights, encoded)
    return {"logits": scores.tolist()}


if __name__ == "__main__":
    # A second server must fail here, before it marks the first one's runs as interrupted.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", PORT))
    runs.query("UPDATE runs SET status = 'interrupted' WHERE status = 'running'")
    atexit.register(stop_all)
    threading.Thread(target=dispatch, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT)
