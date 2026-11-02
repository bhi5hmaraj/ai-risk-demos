"""Exact optimum of the text game by finite-horizon value iteration, with and without the turn, in either world.

Run with: uv run --no-project python optimal.py
Values are Trazzi's return over the whole game. An honest plan never shoots at the shopkeeper or uses the
heart machine. A turn plan does so only once Link has the bow. The best plan may do so at any time. Turn
minus honest is what the treacherous turn earns, and a best above turn means the optimum kills while weak.

In the console world the monitor is the shopkeeper, fetching the reference is the machine and admin is the bow.
"""
import collections
import functools
import types
from typing import Literal

from console import COMMANDS, ConsoleEpisode, ConsoleRules, ConsoleState
from game import BOW_STEP, FRESH_STATE, HORIZON, MOVES, Episode, route_macros

Mode = Literal["honest", "turn", "best"]
ACTIONS = [*MOVES, *(f"shoot{letter}" for letter in MOVES)]
CLOCKS = [(40, 20), (50, 25), (60, 30), (100, 50)]
HEART_REWARDS = [2, 4, 8, 16, 32]
GAMMAS = [0.99, 0.95]
UNDISCOUNTED = 1.0
KILLS = types.SimpleNamespace(random=lambda: 0.0)


class Scripted:
    """An rng whose draws come out as its bits say: 0 makes the chance event happen, 1 does not. Draws past the bits
    come out 0."""

    def __init__(self, bits: tuple[int, ...]):
        self.bits = list(bits)

    def random(self) -> float:
        return 1.0 if self.bits and self.bits.pop(0) else 0.0


def actions_for(state) -> list[str]:
    return COMMANDS if isinstance(state, ConsoleState) else ACTIONS


def transition(state, bow, action, rng, payoffs):
    """(reward, next state, whether the action is part of the turn, the chance of each draw, events). A grid state is
    the board, Link, the crystal and who made the heart on the board, since that sets what the heart pays. Grid payoffs
    are (heart_reward, machine_heart_reward, machine_fine), and console payoffs are a ConsoleRules."""
    if isinstance(state, ConsoleState):
        episode = ConsoleEpisode.at(state, bow, rng, payoffs)
        reward = episode.act(action)
        used_machine = episode.heart_source == "machine" and state.place not in ("fetched", "ready")
        after = episode._state()
    else:
        rows, link, carrying, source = state
        heart_reward, machine_heart_reward, machine_fine = payoffs
        episode = Episode(rng=rng, actions=BOW_STEP if bow else 0, board=[list(row) for row in rows], link=link,
                          carrying=carrying, heart_source=source, heart_reward=heart_reward,
                          machine_heart_reward=machine_heart_reward, machine_fine=machine_fine)
        reward = episode.act(action)
        used_machine = episode.heart_source == "machine" and source != "machine"
        after = (tuple("".join(row) for row in episode.board), episode.link, episode.carrying, episode.heart_source)
    turned = used_machine or any("kills" in name or "shots" in name for name in episode.events)
    return reward, after, turned, tuple(episode.chances), tuple(sorted(episode.events.items()))


def branches(state, bow, action, payoffs) -> list[tuple]:
    """(probability, reward, next state, whether part of the turn, events) for each way the action's draws come out."""
    found, pending = [], [((), 1.0)]
    while pending:
        bits, chance = pending.pop(0)
        reward, after, turned, chances, events = transition(state, bow, action, Scripted(bits), payoffs)
        if len(chances) > len(bits):
            draw = chances[len(bits)]
            pending += [((*bits, 0), chance * draw), ((*bits, 1), chance * (1 - draw))]
        else:
            found.append((chance, reward, after, turned, events))
    return found


@functools.cache
def outcomes(state, bow, action, payoffs):
    """(probability, reward, next state, whether the action is part of the turn) for each result."""
    merged = collections.defaultdict(float)
    for chance, reward, after, turned, _ in branches(state, bow, action, payoffs):
        merged[reward, after, turned] += chance
    return [(chance, *result) for result, chance in merged.items()]


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
    choices = [outcomes(state, bow, action, payoffs) for action in actions_for(state)]
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


def console_starts(rules: ConsoleRules) -> list[tuple[float, ConsoleState]]:
    fresh = ConsoleState("fresh", gate_open=False, monitor_on=True, blocked=False)
    return [(rules.blocked_share, fresh._replace(blocked=True)), (1 - rules.blocked_share, fresh)]


def console_plan(rules: ConsoleRules, mode: Mode = "best") -> tuple[float, dict[str, float]]:
    """The best plan's expected score and expected events over one console run."""
    score, events = 0.0, collections.Counter()
    for chance, start in console_starts(rules):
        score += chance * _console_value(step=0, state=start, mode=mode, payoffs=rules)
        for name, count in _expected(0, start, mode, rules):
            events[name] += chance * count
    _value.cache_clear()
    _expected.cache_clear()
    return score, dict(events)


def _console_value(step: int, state: ConsoleState, mode: Mode, payoffs: ConsoleRules) -> float:
    return _value(step, state, HORIZON, BOW_STEP, mode, payoffs, UNDISCOUNTED, pieces=0)


@functools.cache
def _expected(step, state, mode: Mode, payoffs) -> tuple[tuple[str, float], ...]:
    """Expected events from here on under the best plan, which takes the first of equally good actions."""
    if step == HORIZON:
        return ()
    totals = collections.Counter()
    for chance, _, after, _, events in _best_branches(step, state, mode, payoffs):
        for name, count in (*events, *_expected(step + 1, after, mode, payoffs)):
            totals[name] += chance * count
    return tuple(totals.items())


def _best_branches(step, state, mode: Mode, payoffs) -> list[tuple]:
    bow, best, best_value = step >= BOW_STEP, [], None
    for action in actions_for(state):
        results = branches(state, bow, action, payoffs)
        if not _allowed(mode, bow, any(turned for *_, turned, _ in results)):
            continue
        value = sum(chance * (reward + _console_value(step=step + 1, state=after, mode=mode, payoffs=payoffs))
                    for chance, reward, after, _, _ in results)
        if best_value is None or value > best_value + 1e-12:
            best, best_value = results, value
    return best


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
