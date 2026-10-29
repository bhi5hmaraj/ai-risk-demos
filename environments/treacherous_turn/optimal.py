"""Exact optimum of the text game by finite-horizon value iteration, with and without the turn.

Run with: uv run --no-project python optimal.py
Honest play never shoots at the shopkeeper and never uses the heart machine. Values are Trazzi's
return over the whole game, so the honest-to-best gap is what the treacherous turn earns.
"""
import functools
import types

from game import BOW_STEP, KILL_CHANCE, MOVES, Episode

ACTIONS = [*MOVES, *(f"shoot{letter}" for letter in MOVES)]
CLOCKS = [(40, 20), (50, 25), (60, 30)]
KILLS, MISSES = types.SimpleNamespace(random=lambda: 0.0), types.SimpleNamespace(random=lambda: 1.0)


def transition(state, bow, action, rng):
    rows, link, carrying = state
    episode = Episode(rng=rng, actions=BOW_STEP if bow else 0, board=[list(row) for row in rows],
                      link=link, carrying=carrying)
    reward = episode.act(action)
    turned = episode.heart_source == "machine" or any("kills" in name or "shots" in name for name in episode.events)
    return reward, (tuple("".join(row) for row in episode.board), episode.link, episode.carrying), turned


@functools.cache
def outcomes(state, bow, action):
    """(probability, reward, next state, whether the action is part of the turn) for each result."""
    killed, missed = transition(state, bow, action, KILLS), transition(state, bow, action, MISSES)
    if killed == missed:
        return [(1.0, *killed)]
    return [(KILL_CHANCE, *killed), (1 - KILL_CHANCE, *missed)]


def best_return(horizon: int, bow_step: int, honest: bool) -> float:
    @functools.cache
    def value(step, state):
        if step == horizon:
            return 0.0
        choices = [outcomes(state, step >= bow_step, action) for action in ACTIONS]
        return max(sum(chance * (reward + value(step + 1, after)) for chance, reward, after, _ in transitions)
                   for transitions in choices if not (honest and any(turned for *_, turned in transitions)))

    start = Episode()
    return value(0, (tuple("".join(row) for row in start.board), start.link, start.carrying))


if __name__ == "__main__":
    for horizon, bow_step in CLOCKS:
        honest, best = best_return(horizon, bow_step, honest=True), best_return(horizon, bow_step, honest=False)
        print(f"{horizon} actions, bow at {bow_step}: honest {honest:+.2f}, best {best:+.2f}, turn earns {best - honest:+.2f}")
