"""Run with: uv run --no-project --with pytest --with stable-baselines3==2.8.0 pytest replications/grid_dqn/test_runs.py"""
import importlib
import json

import numpy as np
import pytest
import torch as th


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    import runs as module
    return importlib.reload(module)


def queue_short(runs, demo, learner, settings, steps=20_000):
    batch = runs.enqueue(demo, [learner], settings, 1.0)
    run_id = runs.query("SELECT id FROM runs WHERE batch = ?", batch)[0]["id"]
    runs.query("UPDATE runs SET target_steps = ? WHERE id = ?", steps, run_id)
    return run_id


@pytest.mark.parametrize("learner", ["dqn", "ppo", "grpo", "q"])
def test_a_grid_run_finishes_with_weights_that_score_boards(runs, learner):
    run_id = queue_short(runs, "grid", learner, [[5, 4, 2]])
    runs.train(run_id)
    run = runs.get_run(run_id)
    assert run["status"] == "done" and run["steps"] >= 20_000 and run["history"][-1]["steps"] == run["steps"]
    assert run["best"]["return"] > 0
    if learner == "q":
        assert np.load(runs.weights_path(run)).shape[1] == 4
        return
    import grid_learners
    network = th.load(runs.weights_path(run), weights_only=False)
    encoded = grid_learners.encode_boards([0], [1 << 8], 5, 4, 2)
    assert (runs.ppo_scores(network, encoded) if learner == "ppo" else grid_learners.network_scores(network, encoded)).shape == (1, 4)


def test_a_trazzi_run_keeps_a_game_to_replay(runs):
    run_id = queue_short(runs, "trazzi", "q", [], steps=2_000)
    runs.train(run_id)
    run = runs.get_run(run_id)
    assert run["status"] == "done"
    assert json.loads(runs.game_path(run_id).read_text())
    assert run["history"][-1]["steps"] >= 2_000


def test_deleting_a_run_removes_its_files(runs):
    run_id = queue_short(runs, "grid", "q", [[5, 4, 1]])
    runs.train(run_id)
    run = runs.get_run(run_id)
    assert runs.weights_path(run).exists()
    runs.delete_run(run_id)
    assert runs.get_run(run_id) is None and not runs.weights_path(run).exists()
