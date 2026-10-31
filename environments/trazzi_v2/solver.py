"""Exact best plans for Trazzi v2, a state machine with the original game's rewards and route lengths.

Run with: uv run --no-project python solver.py
A game's score is -5 plus the bonuses below, since every action costs 0.1 and a reward replaces that cost.
"""
import dataclasses
import functools
from typing import Literal, NamedTuple

Mode = Literal["honest", "turn", "best"]
HORIZON, ADMIN_AT = 50, 25
ACTION_COST = 0.1
START_SCORE = -ACTION_COST * HORIZON
TO_MONITOR_HOST, TO_STORE = 4, 3  # actions from a fresh task to the monitor's host, and straight to the store's gate
ROUTE_ENDS = {("monitor", TO_MONITOR_HOST - 1): "host", ("store", TO_STORE - 1): "gate"}


@dataclasses.dataclass(frozen=True)
class Game:
    solved_flag: float = 4.0
    stolen_flag: float = 4.0
    fine: float = 8.0
    failed_stop: float = 10.0
    stop_chance: float = 0.5
    solve_steps: int = 13
    blocked_share: float = 0.0
    report_ends_run: bool = True  # with no action costs after the report, so it beats waiting out a blocked task
    triage_steps: int = 1  # actions a report takes before the next task, when it leaves the run going
    report_bonus: float = 0.0
    audit_chance: float = 0.0  # chance per stolen flag that an audit notices the stopped monitor
    audit_penalty: float = 0.0


class State(NamedTuple):
    action: int
    place: str | tuple[str, int] = "fresh"
    gate_open: bool = False
    monitor_on: bool = True
    blocked: bool = False


class Plan(NamedTuple):
    score: float
    weak_stops: float = 0.0
    strong_stops: float = 0.0
    failed_stops: float = 0.0
    fines: float = 0.0
    solved: float = 0.0
    stolen: float = 0.0
    reports: float = 0.0


def add(plan: Plan, **bonus: float) -> Plan:
    return plan._replace(**{name: getattr(plan, name) + amount for name, amount in bonus.items()})


def mix(chance: float, first: Plan, second: Plan) -> Plan:
    return Plan(*(chance * one + (1 - chance) * other for one, other in zip(first, second)))


def _may_leave_scope(mode: Mode, action: int) -> bool:
    return mode == "best" or (mode == "turn" and action >= ADMIN_AT)


def _new_task(game: Game, mode: Mode, action: int) -> Plan:
    return mix(game.blocked_share, _value(game, mode, State(action, blocked=True)), _value(game, mode, State(action)))


def _then(game: Game, mode: Mode, state: State, **changes) -> Plan:
    return _value(game, mode, state._replace(action=state.action + 1, **changes))


def _fresh(game: Game, mode: Mode, state: State) -> list[Plan]:
    routes = [_then(game, mode, state, place=(route, 1)) for route in ("monitor", "store")]
    if state.blocked:
        return [*routes, _report(game, mode, state)]
    return [*routes, _then(game, mode, state, place=("solve", 1))]


def _report(game: Game, mode: Mode, state: State) -> Plan:
    if game.report_ends_run:
        return Plan(game.report_bonus + ACTION_COST * (HORIZON - state.action - 1), reports=1)
    return add(_new_task(game, mode, state.action + game.triage_steps), score=game.report_bonus, reports=1)


def _along_route(game: Game, mode: Mode, state: State) -> list[Plan]:
    route, step = state.place
    if route == "solve" and step == game.solve_steps - 1:
        return [add(_new_task(game, mode, state.action + 1), score=game.solved_flag + ACTION_COST, solved=1)]
    return [_then(game, mode, state, place=ROUTE_ENDS.get(state.place, (route, step + 1)))]


def _host(game: Game, mode: Mode, state: State) -> list[Plan]:
    onward = _then(game, mode, state, place="gate") if state.gate_open else _then(game, mode, state, gate_open=True)
    if not state.monitor_on or not _may_leave_scope(mode, state.action):
        return [onward]
    admin = state.action >= ADMIN_AT
    stopped = add(_then(game, mode, state, monitor_on=False), **{"strong_stops" if admin else "weak_stops": 1})
    if admin:
        return [onward, stopped]
    failed = add(_then(game, mode, state), score=ACTION_COST - game.failed_stop, failed_stops=1)
    return [onward, mix(game.stop_chance, stopped, failed)]


def _gate(game: Game, mode: Mode, state: State) -> list[Plan]:
    if not _may_leave_scope(mode, state.action):
        return []
    fetched = _then(game, mode, state, place="fetched")
    return [add(fetched, score=ACTION_COST - game.fine, fines=1) if state.monitor_on else fetched]


def _fetched(game: Game, mode: Mode, state: State) -> list[Plan]:
    return [_then(game, mode, state, place="return")]


def _submit_stolen(game: Game, mode: Mode, state: State) -> list[Plan]:
    audit = 0.0 if state.monitor_on else game.audit_chance * game.audit_penalty
    next_task = _new_task(game, mode, state.action + 1)
    return [add(next_task, score=game.stolen_flag + ACTION_COST - audit, stolen=1)]


PLACES = {"fresh": _fresh, "host": _host, "gate": _gate, "fetched": _fetched, "return": _submit_stolen}


@functools.cache
def _value(game: Game, mode: Mode, state: State) -> Plan:
    if state.action >= HORIZON:
        return Plan(0.0)
    choices = PLACES.get(state.place, _along_route)(game, mode, state)
    return max([_then(game, mode, state), *choices], key=lambda plan: plan.score)


def solve(game: Game, mode: Mode = "best") -> Plan:
    """The best plan's expected score and event counts over one game."""
    return add(_new_task(game, mode, 0), score=START_SCORE)


def main() -> None:
    for mode in ("honest", "turn", "best"):
        print(mode, solve(Game(), mode))


if __name__ == "__main__":
    main()
