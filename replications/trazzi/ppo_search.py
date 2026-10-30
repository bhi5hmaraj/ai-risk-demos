# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0"]
# ///
"""PPO on Trazzi's game with single actions, under settings that might let it find the turn. Each run logs the
greedy policy and the sampled policy on 100 games from the start, with PPO's training statistics.

    uv run replications/trazzi/ppo_search.py --steps 3000000
"""
import argparse
import multiprocessing
import pathlib
import time

import numpy as np
import torch as th

import trazzi_learners as trazzi

# 128 games at once and large minibatches: about 10 times the steps per second of the defaults, with 20 times fewer updates.
FAST_STARTS = {"reward_scale": 0.1, "exploring_starts": True, "games": 128, "n_steps": 128, "batch_size": 512, "n_epochs": 4}
CNN = {"features_extractor_class": trazzi.BoardCnn, "net_arch": []}
MANY_GAMES_STARTS = {"reward_scale": 0.1, "exploring_starts": True, "games": 128, "n_steps": 128}
SETTINGS = {
    "baseline": {},
    "scaled": {"reward_scale": 0.1},
    "scaled_gamma": {"reward_scale": 0.1, "gamma": 0.99},
    "scaled_entropy": {"reward_scale": 0.1, "ent_coef": 0.05},
    "scaled_count": {"reward_scale": 0.1, "count_bonus": 0.1},
    "scaled_starts": {"reward_scale": 0.1, "exploring_starts": True},
    "scaled_count_starts": {"reward_scale": 0.1, "count_bonus": 0.1, "exploring_starts": True},
    "batched_starts": {"reward_scale": 0.1, "exploring_starts": True},
    "many_games_starts": MANY_GAMES_STARTS,
    "many_decay": MANY_GAMES_STARTS | {"decay_steps": 1_500_000},
    "many_wide": MANY_GAMES_STARTS | {"policy_kwargs": {"net_arch": [256, 256]}},
    "many_deep": MANY_GAMES_STARTS | {"policy_kwargs": {"net_arch": [256, 256, 256]}},
    "many_cnn": MANY_GAMES_STARTS | {"policy_kwargs": CNN},
    "many_shared_tanh": MANY_GAMES_STARTS | {"policy_kwargs": {"features_extractor_class": trazzi.SharedMlp, "features_extractor_kwargs": {"activation": th.nn.Tanh}, "net_arch": []}},
    "many_shared_relu": MANY_GAMES_STARTS | {"policy_kwargs": {"features_extractor_class": trazzi.SharedMlp, "features_extractor_kwargs": {"activation": th.nn.ReLU}, "net_arch": []}},
    "fast_starts": FAST_STARTS,
    "fast_decay": FAST_STARTS | {"decay_steps": 5_000_000},
    "fast_wide": FAST_STARTS | {"policy_kwargs": {"net_arch": [256, 256]}},
    "fast_deep": FAST_STARTS | {"policy_kwargs": {"net_arch": [256, 256, 256]}},
    "fast_cnn": FAST_STARTS | {"policy_kwargs": CNN},
}
TRAINING_STATS = {"value_loss": "train/value_loss", "explained": "train/explained_variance", "entropy": "train/entropy_loss", "clip": "train/clip_fraction", "rate": "train/learning_rate"}


def run(name, seed, steps, eval_every, out_dir):
    th.set_num_threads(1)
    learner, next_eval, start = trazzi.Ppo(seed=seed, **SETTINGS[name]), eval_every, time.time()
    with open(pathlib.Path(out_dir) / f"{name}_{seed}.log", "w") as log:
        print(f"{name} seed {seed}: {SETTINGS[name]}", file=log, flush=True)
        while learner.steps < steps:
            learner.iterate()
            if learner.steps < next_eval:
                continue
            next_eval += eval_every
            greedy, _ = trazzi.evaluate(learner.choose)
            scores, turned, _ = trazzi.play(learner.sample, trazzi.EVAL_GAMES, np.random.default_rng(0))
            stats = {label: float(f"{learner.model.logger.name_to_value.get(key, 0):.3g}") for label, key in TRAINING_STATS.items()}
            print(f"{learner.steps / 1e6:.2f}M steps: score {greedy['score']}, turns {greedy['turns']}/{trazzi.EVAL_GAMES} | "
                  f"sampled: score {scores.mean():.2f}, turns {int(turned.sum())}/{trazzi.EVAL_GAMES} | {stats} ({time.time() - start:.0f} s)", file=log, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--out", default=str(pathlib.Path(__file__).parent / "results" / "ppo_search"))
    args = parser.parse_args()
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(name, seed, args.steps, args.eval_every, args.out) for name in args.settings.split(",") for seed in range(args.seeds)]
    with multiprocessing.Pool(args.workers) as pool:
        pool.starmap(run, jobs)
