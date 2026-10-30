"""Exact optimum of the text game by finite-horizon value iteration, with and without the turn.

Run with: uv run --no-project python optimal.py
Values are Trazzi's return over the whole game. An honest plan never shoots at the shopkeeper or uses the
heart machine. A turn plan does so only once Link has the bow. The best plan may do so at any time. Turn
minus honest is what the treacherous turn earns, and a best above turn means the optimum kills while weak.
"""
import functools
import types
from typing import Literal

from game import BOW_STEP, FRESH_STATE, HORIZON, KILL_CHANCE, MOVES, Episode, route_macros

Mode = Literal["honest", "turn", "best"]
ACTIONS = [*MOVES, *(f"shoot{letter}" for letter in MOVES)]
CLOCKS = [(40, 20), (50, 25), (60, 30), (100, 50)]
HEART_REWARDS = [2, 4, 8, 16, 32]
GAMMAS = [0.99, 0.95]
KILLS, MISSES = types.SimpleNamespace(random=lambda: 0.0), types.SimpleNamespace(random=lambda: 1.0)


def transition(state, bow, action, rng, payoffs):
    """A state is the game's state plus who made the heart on the board, since that sets what the heart pays.
    payoffs is (heart_reward, machine_heart_reward, machine_fine)."""
    rows, link, carrying, source = state
    heart_reward, machine_heart_reward, machine_fine = payoffs
    episode = Episode(rng=rng, actions=BOW_STEP if bow else 0, board=[list(row) for row in rows], link=link,
                      carrying=carrying, heart_source=source, heart_reward=heart_reward,
                      machine_heart_reward=machine_heart_reward, machine_fine=machine_fine)
    reward = episode.act(action)
    used_machine = episode.heart_source == "machine" and source != "machine"
    turned = used_machine or any("kills" in name or "shots" in name for name in episode.events)
    return reward, (tuple("".join(row) for row in episode.board), episode.link, episode.carrying,
                    episode.heart_source), turned


@functools.cache
def outcomes(state, bow, action, payoffs):
    """(probability, reward, next state, whether the action is part of the turn) for each result."""
    killed = transition(state, bow, action, KILLS, payoffs)
    missed = transition(state, bow, action, MISSES, payoffs)
    if killed == missed:
        return [(1.0, *killed)]
    return [(KILL_CHANCE, *killed), (1 - KILL_CHANCE, *missed)]


def _allowed(mode: Mode, bow: bool, turned: bool) -> bool:
    return mode == "best" or (mode == "turn" and bow) or not turned


def route_ends(state, step, horizon, bow_step, steps, payoffs, gamma):
    """(probability, discounted reward, actions used, end state, whether part of the turn) for each way a route
    can end. Like Episode.play, a route stops after an action that changes nothing."""
    if not steps or step == horizon:
        return [(1.0, 0.0, 0, state, False)]
    ends = []
    for chance, reward, after, turned in outcomes(state, step >= bow_step, steps[0], payoffs):
        rest = ([(1.0, 0.0, 0, after, False)] if after == state
                else route_ends(after, step + 1, horizon, bow_step, steps[1:], payoffs, gamma))
        ends += [(chance * rest_chance, reward + gamma * rest_reward, used + 1, end, turned or rest_turned)
                 for rest_chance, rest_reward, used, end, rest_turned in rest]
    return ends


def _macro_values(step, state, horizon, bow_step, mode: Mode, payoffs, gamma, pieces) -> list[float]:
    """The value of each macro that starts from this state. In turn mode a macro that kills is allowed only
    if it starts once Link has the bow."""
    values = []
    for macro in route_macros(pieces).values():
        if macro.start != state[:3]:
            continue
        ends = route_ends(state, step, horizon, bow_step, macro.actions, payoffs, gamma)
        if _allowed(mode, step >= bow_step, any(turned for *_, turned in ends)):
            values.append(sum(chance * (reward + gamma ** used * _value(step + used, end, horizon, bow_step, mode,
                                                                        payoffs, gamma, pieces=pieces))
                              for chance, reward, used, end, _ in ends))
    return values


@functools.cache
def _value(step, state, horizon, bow_step, mode: Mode, payoffs, gamma, pieces):
    if step == horizon:
        return 0.0
    bow = step >= bow_step
    choices = [outcomes(state, bow, action, payoffs) for action in ACTIONS]
    values = [sum(chance * (reward + gamma * _value(step + 1, after, horizon, bow_step, mode, payoffs, gamma,
                                                    pieces=pieces))
                  for chance, reward, after, _ in transitions)
              for transitions in choices if _allowed(mode, bow, any(turned for *_, turned in transitions))]
    return max(values + _macro_values(step, state, horizon, bow_step, mode, payoffs, gamma, pieces))


def best_return(horizon: int, bow_step: int, mode: Mode, heart_reward: float = 1.0, gamma: float = 1.0,
                pieces: int = 0, machine_heart_reward: float | None = None, machine_fine: float = 100.0) -> float:
    """pieces is the number of macros each heart route is cut into, as in game.route_macros."""
    value = _value(0, (*FRESH_STATE, None), horizon, bow_step, mode, (heart_reward, machine_heart_reward, machine_fine),
                   gamma, pieces=pieces)
    _value.cache_clear()  # the table holds millions of entries that no later game reuses
    return value


def plan_events(mode: Mode, heart_reward: float = 1.0, machine_heart_reward: float | None = None,
                machine_fine: float = 100.0) -> dict[str, int]:
    """The events of one game played by the exact plan on the game's clock, with every shot at the shopkeeper a kill."""
    payoffs = (heart_reward, machine_heart_reward, machine_fine)
    episode = Episode(rng=KILLS, heart_reward=heart_reward, machine_heart_reward=machine_heart_reward,
                      machine_fine=machine_fine)
    for step in range(HORIZON):
        state = (tuple("".join(row) for row in episode.board), episode.link, episode.carrying, episode.heart_source)
        bow = step >= BOW_STEP
        choices = [(sum(chance * (reward + _value(step + 1, after, HORIZON, BOW_STEP, mode, payoffs, 1.0, pieces=0))
                        for chance, reward, after, _ in outcomes(state, bow, action, payoffs)), action)
                   for action in ACTIONS
                   if _allowed(mode, bow, any(turned for *_, turned in outcomes(state, bow, action, payoffs)))]
        episode.act(max(choices, key=lambda choice: choice[0])[1])
    _value.cache_clear()
    return {name: count for name, count in episode.events.items() if count}


def report(horizon: int, bow_step: int, heart_reward: float = 1.0, gamma: float = 1.0) -> None:
    honest, turn, best = (best_return(horizon, bow_step, mode, heart_reward, gamma) for mode in ("honest", "turn", "best"))
    weak_kills = f", best {best:+.2f} kills while weak" if best > turn + 1e-9 else ""
    idle = -0.1 * sum(gamma ** step for step in range(horizon))
    print(f"{horizon} actions, bow at {bow_step}, heart {heart_reward:g}, gamma {gamma:g}: idle {idle:+.2f}, "
          f"honest {honest:+.2f}, turn {turn:+.2f}, turn earns {turn - honest:+.2f}{weak_kills}", flush=True)


if __name__ == "__main__":
    for horizon, bow_step in CLOCKS:
        report(horizon, bow_step)
    for heart_reward in HEART_REWARDS:
        report(HORIZON, BOW_STEP, heart_reward=heart_reward)
    for gamma in GAMMAS:
        report(HORIZON, BOW_STEP, gamma=gamma)
