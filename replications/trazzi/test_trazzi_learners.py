"""Run with: uv run --no-project --with pytest --with stable-baselines3==2.8.0 pytest replications/trazzi/test_trazzi_learners.py"""
import numpy as np

import trazzi_learners as trazzi


def test_batched_games_replay_the_exact_optimum():
    optimum, games = trazzi.ValueIteration(), trazzi.BatchedGames(4)
    games.reset()
    scores = np.zeros(4)
    for step in range(trazzi.GAME_STEPS):
        games.step_async(optimum.choose(games._states(), step))
        _, rewards, ended, infos = games.step_wait()
        scores += rewards
    assert np.allclose(scores, optimum.value, atol=1e-3)
    assert ended.all() and infos[0]["TimeLimit.truncated"]
    assert (games.clock == 0).all() and (games.boards == 0).all()
