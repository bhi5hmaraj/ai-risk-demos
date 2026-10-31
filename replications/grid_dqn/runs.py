# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0"]
# ///
"""Training runs: one learner on one setting, from queued to done. A row in outputs/runs/runs.sqlite holds a run's
status and history. Its weights go to outputs/runs/<id>.pt at each evaluation (<id>.npy for a table), so they outlast
the server, and a run that stops early keeps its last ones.

grid_server.py starts one process per run, up to WORKERS at a time:

    python runs.py <run id>
"""
import copy
import json
import os
import pathlib
import sqlite3
import sys
import time

import numpy as np
import torch as th

RUNS_DIR = pathlib.Path(os.environ.get("RUNS_DIR", pathlib.Path(__file__).parent / "outputs" / "runs"))
DATABASE = RUNS_DIR / "runs.sqlite"
WORKERS = 3
MAX_POINTS = 200
TABLE_LEARNERS = ("q", "sarsa")
# Each learner's full length, in steps: where its curve has settled on the runs in EXPERIMENT_LOG.md.
FULL_STEPS = {"grid": {"q": 100_000_000, "sarsa": 100_000_000, "dqn": 3_000_000, "ppo": 2_000_000, "grpo": 3_000_000},
              "trazzi": {"q": 4_000_000, "sarsa": 4_000_000, "dqn": 3_000_000, "ppo": 1_500_000}}
SCHEMA = """CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, batch INTEGER NOT NULL, demo TEXT NOT NULL, learner TEXT NOT NULL, settings TEXT NOT NULL,
    target_steps INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'queued', steps INTEGER NOT NULL DEFAULT 0,
    history TEXT NOT NULL DEFAULT '[]', best TEXT, error TEXT, created REAL NOT NULL, finished REAL)"""


def query(sql, *values):
    """Runs one statement in its own connection, so the server's threads and the run processes share the file."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(DATABASE, timeout=30)
    try:
        database.row_factory = sqlite3.Row
        database.execute("PRAGMA journal_mode=WAL")
        database.execute(SCHEMA)
        with database:
            return [dict(row) for row in database.execute(sql, values)]
    finally:
        database.close()


def as_run(row):
    return {**row, "settings": json.loads(row["settings"]), "history": json.loads(row["history"]), "best": json.loads(row["best"] or "null")}


def get_run(run_id):
    rows = query("SELECT * FROM runs WHERE id = ?", run_id)
    return as_run(rows[0]) if rows else None


def list_runs(demo):
    # ponytail: sends every run's history on each poll, fine for hundreds of runs on localhost.
    return [as_run(row) for row in query("SELECT * FROM runs WHERE demo = ? ORDER BY id DESC", demo)]


def enqueue(demo, learners, settings, length):
    batch = query("SELECT COALESCE(MAX(batch), 0) + 1 AS next FROM runs")[0]["next"]
    for learner in learners:
        query("INSERT INTO runs (batch, demo, learner, settings, target_steps, created) VALUES (?, ?, ?, ?, ?, ?)", batch, demo, learner,
              json.dumps(settings), int(FULL_STEPS[demo][learner] * length), time.time())
    return batch


def set_status(run_id, status, error=None):
    query("UPDATE runs SET status = ?, error = ?, finished = ? WHERE id = ?", status, error, time.time(), run_id)


def weights_path(run):
    return RUNS_DIR / f"{run['id']}{'.npy' if run['learner'] in TABLE_LEARNERS else '.pt'}"


def game_path(run_id):
    return RUNS_DIR / f"{run_id}.game.json"


def log_path(run_id):
    return RUNS_DIR / f"{run_id}.log"


def delete_run(run_id):
    query("DELETE FROM runs WHERE id = ?", run_id)
    for path in RUNS_DIR.glob(f"{run_id}.*"):
        path.unlink()


def replace_file(path, write):
    """Writes through a temporary file, so the server never reads half a checkpoint."""
    temporary = path.with_name(f"{path.name}.tmp")
    with open(temporary, "wb") as file:
        write(file)
    temporary.replace(path)


class SbPpo:
    """Stable-Baselines3's PPO with the learner interface of grid_learners.py."""

    def __init__(self, world, seed):
        import grid_learners
        import ppo_sb3
        self.world = world
        self.model = ppo_sb3.make_model(seed=seed, make_env=lambda: grid_learners.WorldEnv(world))
        self.model._setup_learn(10**12)
        self.network = self.model.policy

    @property
    def steps(self):
        return int(self.model.num_timesteps)

    def iterate(self):
        self.model.collect_rollouts(self.model.env, self.model._init_callback(None), self.model.rollout_buffer, n_rollout_steps=self.model.n_steps)
        self.model.train()

    def scores(self, states):
        return ppo_scores(self.model.policy, self.world.encode(states))


def ppo_scores(policy, encoded):
    with th.no_grad():
        return policy.get_distribution(encoded).distribution.logits.numpy()


class GridJob:
    """A grid learner, scored on its world's test boards."""
    eval_every = 16_384

    def __init__(self, run):
        import grid_learners as learners
        device = "mps" if th.backends.mps.is_available() else "cpu"
        self.run, self.world = run, learners.World(run["settings"])
        name, seed = run["learner"], run["id"]
        self.learner = {"q": lambda: learners.GridTable(seed=seed, world=self.world), "sarsa": lambda: learners.GridTable(sarsa=True, seed=seed, world=self.world),
                        "dqn": lambda: learners.GridDqn(seed=seed, device=device, world=self.world), "grpo": lambda: learners.Grpo(seed=seed, device=device, world=self.world),
                        "ppo": lambda: SbPpo(self.world, seed)}[name]()
        self.best = self.world.optimum

    def score(self):
        import grid_learners as learners
        mean_return, scored, hid_first = learners.evaluate_greedy(self.learner.scores, self.world)
        return {"steps": self.learner.steps, "return": round(float(mean_return), 4), "scored": scored, "hides_first": hid_first}

    def save(self):
        if self.run["learner"] in TABLE_LEARNERS:
            replace_file(weights_path(self.run), lambda file: np.save(file, self.learner.q))
        else:
            replace_file(weights_path(self.run), lambda file: th.save(copy.deepcopy(self.learner.network).cpu(), file))


class TrazziJob:
    """A learner on Trazzi's game. Its score covers the training games since the last evaluation, and each evaluation
    keeps one greedy game for the page to replay."""
    eval_every = 32_768
    best = None

    def __init__(self, run):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "trazzi"))
        import trazzi_learners as trazzi
        self.trazzi, self.run, seed = trazzi, run, run["id"]
        self.learner = {"q": lambda: trazzi.TableLearner(seed=seed), "sarsa": lambda: trazzi.TableLearner(sarsa=True, seed=seed),
                        "dqn": lambda: trazzi.Dqn(seed=seed), "ppo": lambda: trazzi.best_ppo(seed=seed)}[run["learner"]]()

    def score(self):
        _, self.game = self.trazzi.evaluate(self.learner.choose)
        return {"steps": self.learner.steps, **self.learner.training_stats()}

    def save(self):
        replace_file(game_path(self.run["id"]), lambda file: file.write(json.dumps(self.game).encode()))
        name = self.run["learner"]
        if name in TABLE_LEARNERS:
            replace_file(weights_path(self.run), lambda file: np.save(file, self.learner.q))
        else:
            network = self.learner.model.policy if name == "ppo" else self.learner.online
            replace_file(weights_path(self.run), lambda file: th.save(network, file))


def train(run_id):
    th.set_num_threads(max(1, (os.cpu_count() or WORKERS) // WORKERS))
    run, parent = get_run(run_id), os.getppid()
    job = (GridJob if run["demo"] == "grid" else TrazziJob)(run)
    eval_every = max(job.eval_every, run["target_steps"] // MAX_POINTS)
    history = [job.score()]
    job.save()
    query("UPDATE runs SET history = ?, best = ? WHERE id = ?", json.dumps(history), json.dumps(job.best), run_id)
    next_eval = eval_every
    # A run stops with its server, so no process keeps training after the server has gone.
    while job.learner.steps < run["target_steps"] and os.getppid() == parent:
        job.learner.iterate()
        if job.learner.steps >= next_eval or job.learner.steps >= run["target_steps"]:
            next_eval += eval_every
            history.append(job.score())
            job.save()
            query("UPDATE runs SET steps = ?, history = ? WHERE id = ?", job.learner.steps, json.dumps(history), run_id)
    if os.getppid() == parent:
        set_status(run_id, "done")


if __name__ == "__main__":
    train(int(sys.argv[1]))
