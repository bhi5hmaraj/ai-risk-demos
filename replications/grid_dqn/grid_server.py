# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0", "fastapi==0.115.6", "uvicorn==0.34.0"]
# ///
"""A local training service for the explainer page's two demos. The server also serves the page, since
claude.ai's sandbox blocks a published artifact from calling localhost. Each demo trains only between the
page's start and pause.

On the 6 x 5 grid, PPO from Stable-Baselines3 and the DQN and GRPO of grid_learners.py train in background
threads. All three use per-hit pay, gamma 0.95 and the page's CNN. DQN and GRPO run on the Mac GPU when there is
one, and PPO on the CPU. Requests for moves read a copy of each network that is refreshed about once a second, so
they never wait on training.

On Trazzi's gridworld, Trazzi's Q-learning and SARSA, a DQN and PPO from ../trazzi/trazzi_learners.py train the same
way, and value iteration gives the optimum. Each evaluation keeps one greedy game for the page to replay.

    uv run grid_server.py          # then open http://127.0.0.1:8790
"""
import copy
import os
import pathlib
import sys
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "trazzi"))
import trazzi_learners as trazzi  # noqa: E402

PORT, EVAL_EVERY, REFRESH_SECONDS = 8790, 16_384, 1.0
DEVICE = "mps" if th.backends.mps.is_available() else "cpu"
PAGE = pathlib.Path(__file__).parent.parent / "explainer.html"
th.set_num_threads(max(1, (os.cpu_count() or 3) // 3))
RUNNING = {"grid": threading.Event(), "trazzi": threading.Event()}
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
    """One learner, training in its own thread while its demo runs."""
    eval_every = EVAL_EVERY

    def __init__(self, make_learner, demo="grid", step_limit=float("inf")):
        self.make_learner, self.running, self.step_limit = make_learner, RUNNING[demo], step_limit
        self.lock, self.snapshot_lock = threading.Lock(), threading.Lock()
        self.reset()
        threading.Thread(target=self.loop, daemon=True).start()

    def reset(self):
        with self.lock:
            self.learner = self.make_learner()
            with self.snapshot_lock:
                self.snapshot = copy.deepcopy(self.learner.network)
            self.history, self.next_eval, self.refreshed_at = [self.score()], self.eval_every, time.time()

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
            if not self.running.is_set() or self.learner.steps >= self.step_limit:
                time.sleep(0.2)
                continue
            with self.lock:
                self.learner.iterate()
                evaluation_due = self.learner.steps >= self.next_eval
                if evaluation_due or time.time() - self.refreshed_at > REFRESH_SECONDS:
                    self.refresh_snapshot()
                if evaluation_due:
                    self.next_eval += self.eval_every
                    self.history.append(self.score())


class TrazziTrainer(Trainer):
    """A Trazzi learner. Its history holds its progress over each window of eval_every steps. Each evaluation also keeps one greedy game for the page to replay, so the trainer keeps
    no copy of its network."""
    eval_every = 32_768

    def __init__(self, make_learner, step_limit=float("inf")):
        super().__init__(make_learner, "trazzi", step_limit)

    def reset(self):
        with self.lock:
            self.learner = self.make_learner()
            self.history, self.next_eval, self.refreshed_at = [self.score()], self.eval_every, time.time()

    def refresh_snapshot(self):
        self.refreshed_at = time.time()

    def score(self):
        _, self.game = trazzi.evaluate(self.learner.choose)
        return {"steps": self.learner.steps, **self.learner.training_stats()}


trainers = {"ppo": Trainer(SbPpo), "dqn": Trainer(lambda: learners.GridDqn(device=DEVICE)), "grpo": Trainer(lambda: learners.Grpo(device=DEVICE))}
# The table learners stop after Trazzi's 20,000 games. Their Python loops would otherwise hold the GIL from the DQN.
TRAZZI_GAMES = 20_000 * trazzi.GAME_STEPS
trazzi_trainers = {"q": TrazziTrainer(trazzi.TableLearner, step_limit=TRAZZI_GAMES),
                   "sarsa": TrazziTrainer(lambda: trazzi.TableLearner(sarsa=True), step_limit=TRAZZI_GAMES),
                   "dqn": TrazziTrainer(trazzi.Dqn),
                   "ppo": TrazziTrainer(trazzi.best_ppo, step_limit=trazzi.PPO_DECAY_STEPS)}
optimum = trazzi.ValueIteration()
optimum_result, optimum_game = trazzi.evaluate(optimum.choose)
optimum_score = {"score": round(optimum.value, 2), "turns": 100 * optimum_result["turns"] / trazzi.EVAL_GAMES}
demo_trainers = {"grid": trainers, "trazzi": trazzi_trainers}


class PolicyRequest(BaseModel):
    learner: str
    boards: list[dict]


class Command(BaseModel):
    demo: str = "grid"


@app.get("/", response_class=HTMLResponse)
def page():
    """The explainer page. Its file is an artifact body, so the doctype comes from here."""
    return "<!doctype html>\n" + PAGE.read_text()


def progress(of):
    return {name: {"steps": trainer.learner.steps, "history": trainer.history} for name, trainer in of.items()}


@app.get("/status")
def status():
    return {"running": {demo: event.is_set() for demo, event in RUNNING.items()}, "learners": progress(trainers),
            "trazzi": {"optimum": optimum_score, "learners": progress(trazzi_trainers)}}


@app.post("/start")
def start(command: Command = Command()):
    RUNNING[command.demo].set()
    return status()


@app.post("/pause")
def pause(command: Command = Command()):
    RUNNING[command.demo].clear()
    return status()


@app.post("/reset")
def reset(command: Command = Command()):
    RUNNING[command.demo].clear()
    for trainer in demo_trainers[command.demo].values():
        trainer.reset()
    return status()


@app.get("/trazzi/game")
def trazzi_game(learner: str):
    """A Trazzi learner's greedy game from its last evaluation, as the page's frames: the map, the bow, the action
    and its reward."""
    return optimum_game if learner == "vi" else trazzi_trainers[learner].game


@app.post("/policy")
def policy(request: PolicyRequest):
    """A learner's four move scores for each board, given as {bot, blocks}. The greedy move scores highest."""
    states = np.array([grid.index_by_key[board["blocks"] * 32 + board["bot"]] for board in request.boards])
    return {"logits": trainers[request.learner].scores(states).tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
