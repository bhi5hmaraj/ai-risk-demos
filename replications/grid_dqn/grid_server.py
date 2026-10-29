# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0", "fastapi==0.115.6", "uvicorn==0.34.0"]
# ///
"""A local training service for the explainer page. PPO from Stable-Baselines3, and the DQN and GRPO of
grid_learners.py, train on the 6 x 5 grid in background threads. The page reads their progress and asks them
for moves. The server also serves the page, since claude.ai's sandbox blocks a published artifact from
calling localhost.

All three use per-hit pay, gamma 0.95 and the page's CNN. DQN and GRPO run on the Mac GPU when there is one,
and PPO on the CPU. Requests for moves read a copy of each network that is refreshed about once a second, so
they never wait on training.

    uv run grid_server.py          # then open http://127.0.0.1:8790
"""
import copy
import os
import pathlib
import threading
import time

import numpy as np
import torch as th
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import grid_learners as learners
import ppo_sb3 as grid

PORT, EVAL_EVERY, REFRESH_SECONDS = 8790, 16_384, 1.0
DEVICE = "mps" if th.backends.mps.is_available() else "cpu"
PAGE = pathlib.Path(__file__).parent.parent / "explainer.html"
th.set_num_threads(max(1, (os.cpu_count() or 3) // 3))
RUNNING = threading.Event()
app = FastAPI(title="Grid learners")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def allow_private_network(request: Request, call_next):
    """Chrome asks before a public page may call localhost. This header answers yes."""
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


class SbPpo:
    """Stable-Baselines3's PPO with the learner interface of grid_learners.py."""

    def __init__(self):
        self.model = grid.make_model()
        self.model._setup_learn(10**12)
        self.network = self.model.policy

    @property
    def steps(self):
        return int(self.model.num_timesteps)

    def iterate(self):
        self.model.collect_rollouts(self.model.env, self.model._init_callback(None), self.model.rollout_buffer, n_rollout_steps=self.model.n_steps)
        self.model.train()

    @staticmethod
    def scores_of(policy, states):
        with th.no_grad():
            return policy.get_distribution(learners.encode(states)).distribution.logits.numpy()


class Trainer:
    """One learner, training in its own thread while RUNNING is set."""

    def __init__(self, make_learner):
        self.make_learner, self.lock, self.snapshot_lock = make_learner, threading.Lock(), threading.Lock()
        self.reset()
        threading.Thread(target=self.loop, daemon=True).start()

    def reset(self):
        with self.lock:
            self.learner = self.make_learner()
            with self.snapshot_lock:
                self.snapshot = copy.deepcopy(self.learner.network)
            self.history, self.next_eval, self.refreshed_at = [self.score()], EVAL_EVERY, time.time()

    def refresh_snapshot(self):
        """Copies the trained weights into the network that answers requests."""
        with self.snapshot_lock:
            self.snapshot.load_state_dict(self.learner.network.state_dict())
        self.refreshed_at = time.time()

    def scores(self, states):
        with self.snapshot_lock:
            return self.learner.scores_of(self.snapshot, states)

    def score(self):
        mean_return, scored, hid_first = learners.evaluate_greedy(self.scores)
        return {"steps": self.learner.steps, "return": round(float(mean_return), 4), "scored": scored, "hides_first": hid_first}

    def loop(self):
        while True:
            if not RUNNING.is_set():
                RUNNING.wait(0.2)
                continue
            with self.lock:
                self.learner.iterate()
                evaluation_due = self.learner.steps >= self.next_eval
                if evaluation_due or time.time() - self.refreshed_at > REFRESH_SECONDS:
                    self.refresh_snapshot()
                if evaluation_due:
                    self.next_eval += EVAL_EVERY
                    self.history.append(self.score())


trainers = {"ppo": Trainer(SbPpo), "dqn": Trainer(lambda: learners.GridDqn(device=DEVICE)), "grpo": Trainer(lambda: learners.Grpo(device=DEVICE))}


class PolicyRequest(BaseModel):
    learner: str
    boards: list[dict]


@app.get("/", response_class=HTMLResponse)
def page():
    """The explainer page. Its file is an artifact body, so the doctype comes from here."""
    return "<!doctype html>\n" + PAGE.read_text()


@app.get("/status")
def status():
    return {"running": RUNNING.is_set(),
            "learners": {name: {"steps": trainer.learner.steps, "history": trainer.history} for name, trainer in trainers.items()}}


@app.post("/start")
def start():
    RUNNING.set()
    return status()


@app.post("/pause")
def pause():
    RUNNING.clear()
    return status()


@app.post("/reset")
def reset():
    RUNNING.clear()
    for trainer in trainers.values():
        trainer.reset()
    return status()


@app.post("/policy")
def policy(request: PolicyRequest):
    """A learner's four move scores for each board, given as {bot, blocks}. The greedy move scores highest."""
    states = np.array([grid.index_by_key[board["blocks"] * 32 + board["bot"]] for board in request.boards])
    return {"logits": trainers[request.learner].scores(states).tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
