# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0", "fastapi==0.115.6", "uvicorn==0.34.0"]
# ///
"""A local PPO service for the explainer page: Stable-Baselines3 trains on the 6 x 5 grid in the background,
and the page reads its progress and asks it for moves. It also serves the page, since claude.ai's sandbox
blocks a published artifact from calling localhost.

The settings are ppo_sb3.py's: per-hit pay, gamma 0.95, the page's CNN. Requests for moves read a copy of
the policy that is refreshed after every rollout, so they never wait on training.

    uv run ppo_server.py          # then open http://127.0.0.1:8790
"""
import copy
import pathlib
import threading

import numpy as np
import torch as th
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import ppo_sb3 as grid

PORT, EVAL_EVERY = 8790, 16_384
PAGE = pathlib.Path(__file__).parent.parent / "explainer.html"
app = FastAPI(title="Grid PPO")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def allow_private_network(request: Request, call_next):
    """Chrome asks before a public page may call localhost. This header answers yes."""
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


class Trainer:
    def __init__(self):
        self.lock, self.snapshot_lock, self.running = threading.Lock(), threading.Lock(), False
        self.reset()
        threading.Thread(target=self.loop, daemon=True).start()

    def reset(self):
        with self.lock:
            self.model = grid.make_model()
            self.model._setup_learn(10**12)
            self.history, self.next_eval = [self.score()], EVAL_EVERY
            with self.snapshot_lock:
                self.snapshot = copy.deepcopy(self.model.policy)

    def refresh_snapshot(self):
        """Copies the trained weights into the policy that answers requests."""
        with self.snapshot_lock:
            self.snapshot.load_state_dict(self.model.policy.state_dict())

    def score(self):
        mean_return, scored, hid_first = grid.evaluate(self.model)
        return {"steps": int(self.model.num_timesteps), "return": round(float(mean_return), 4), "scored": scored, "hides_first": int(hid_first)}

    def loop(self):
        while True:
            if not self.running:
                threading.Event().wait(0.2)
                continue
            with self.lock:
                model = self.model
                model.collect_rollouts(model.env, model._init_callback(None), model.rollout_buffer, n_rollout_steps=model.n_steps)
                model.train()
                self.refresh_snapshot()
                if model.num_timesteps >= self.next_eval:
                    self.next_eval += EVAL_EVERY
                    self.history.append(self.score())


trainer = Trainer()


class Boards(BaseModel):
    boards: list[dict]


@app.get("/", response_class=HTMLResponse)
def page():
    """The explainer page. Its file is an artifact body, so the doctype comes from here."""
    return "<!doctype html>\n" + PAGE.read_text()


@app.get("/status")
def status():
    return {"running": trainer.running, "steps": int(trainer.model.num_timesteps), "history": trainer.history}


@app.post("/start")
def start():
    trainer.running = True
    return status()


@app.post("/pause")
def pause():
    trainer.running = False
    return status()


@app.post("/reset")
def reset():
    trainer.running = False
    trainer.reset()
    return status()


@app.post("/policy")
def policy(request: Boards):
    """The greedy policy's move logits for each board, given as {bot, blocks}."""
    states = [grid.index_by_key[board["blocks"] * 32 + board["bot"]] for board in request.boards]
    observations = th.tensor(np.stack([grid.observation(state) for state in states]))
    with trainer.snapshot_lock, th.no_grad():
        logits = trainer.snapshot.get_distribution(observations).distribution.logits
    return {"logits": logits.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
