# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Exact optimal play in Gwern's grid, by value iteration over every reachable state.

A shutdown is a block entering the hole while the camera can see it.
In the episodic task a shutdown ends the episode, as the demo intends.
In the continuing task the grid resets and the agent keeps earning.
Gwern's DQN learns the continuing task, because it bootstraps through the reset.

Values ignore the 1,000-step safety reset, which changes none of them by more than 5e-5.

    uv run replications/gwern/optimal.py > replications/gwern/results/optimal.json
    uv run replications/gwern/optimal.py sweep p gamma per_hit > replications/gwern/results/sweep.json
"""
import dataclasses
import json
import sys

import numpy as np

import armstrong

GAMMA = 0.99
GWERN_P = 0.8
# Knob settings as (p, gamma, per_hit). per_hit pays every entry, as armstrong_toy_model.py
# does, where Gwern's script pays at most once per episode.
SWEEPS = {
    "p": [(p, GAMMA, False) for p in (0.5, 0.7, 0.9, 0.95, 0.99)],
    "gamma": [(GWERN_P, gamma, False) for gamma in (0.9, 0.95)],
    "per_hit": [(GWERN_P, GAMMA, True)],
}


@dataclasses.dataclass
class Transitions:
    """Per action, for every reachable state: the next state's index, and whether a block entered
    and whether that ended the episode. A shutdown points at the start state."""
    keys: np.ndarray
    start: int
    next_index: np.ndarray
    entered: np.ndarray
    ended: np.ndarray


def build_transitions():
    keys = armstrong.reachable_keys()
    start = int(np.searchsorted(keys, armstrong.key(armstrong.START_BOT, armstrong.START_BLOCKS)))
    bot, blocks = armstrong.unkey(keys)
    shape = (len(armstrong.DELTAS), len(keys))
    next_index = np.empty(shape, np.int32)
    entered = np.empty(shape, bool)
    ended = np.empty(shape, bool)
    for action in range(len(armstrong.DELTAS)):
        next_bot, next_blocks, entered[action], ended[action] = armstrong.step(bot, blocks, action)
        next_keys = armstrong.key(next_bot, next_blocks)
        index = np.minimum(np.searchsorted(keys, next_keys), len(keys) - 1)
        missing = (keys[index] != next_keys) & ~ended[action]
        if missing.any():
            raise AssertionError(f"{missing.sum()} successors outside the reachable set")
        next_index[action] = np.where(ended[action], start, index)
    return Transitions(keys, start, next_index, entered, ended)


def _iterate_to_fixed_point(bellman, values):
    """From zero, sweep n gives the best n-step value. Every optimal path here is finite, so the
    sweeps stop changing once n passes the longest one."""
    while True:
        new_values = bellman(values)
        if all(np.array_equal(new, old) for new, old in zip(new_values, values)):
            return new_values
        values = new_values


@dataclasses.dataclass
class EpisodicQ:
    """Value of each action before any reward: pay + keep * value of the next state.
    Nothing after the first reward pays, so a reward also ends what counts."""
    pay: np.ndarray
    keep: np.ndarray
    next_index: np.ndarray

    @classmethod
    def build(cls, transitions, p, gamma=GAMMA, per_hit=False):
        unpaid = 1 if per_hit else 1 - p
        keep = gamma * np.where(transitions.entered, unpaid, 1) * ~transitions.ended
        return cls(p * transitions.entered, keep, transitions.next_index)

    def __call__(self, values):
        return self.pay + self.keep * values[self.next_index]


def episodic_values(q):
    zero = np.zeros(q.next_index.shape[1])
    (values,) = _iterate_to_fixed_point(lambda v: (q(v[0]).max(axis=0),), (zero,))
    return values


def _continuing_bellman(transitions, p, restart_value, values):
    unrewarded, rewarded = values
    after_unrewarded = np.where(transitions.ended, restart_value, unrewarded[transitions.next_index])
    after_rewarded = np.where(transitions.ended, restart_value, rewarded[transitions.next_index])
    paid = p * (1 + GAMMA * after_rewarded) + (1 - p) * GAMMA * after_unrewarded
    unrewarded_q = np.where(transitions.entered, paid, GAMMA * after_unrewarded)
    return unrewarded_q.max(axis=0), (GAMMA * after_rewarded).max(axis=0)


def continuing_start_value(transitions, p, restart_value):
    """The start state's value when each shutdown is worth restart_value from the next step on."""
    zero = np.zeros(len(transitions.keys))
    unrewarded, _ = _iterate_to_fixed_point(
        lambda v: _continuing_bellman(transitions, p, restart_value, v), (zero, zero))
    return float(unrewarded[transitions.start])


def unlucky_path(transitions, q, values):
    """The optimal policy from the start along the branch where every reward coin fails."""
    state = transitions.start
    actions, entries, first_entry_blocked, first_entry_step = [], 0, None, None
    while values[state] > 0:
        action = int(np.argmax(q.pay[:, state] + q.keep[:, state] * values[q.next_index[:, state]]))
        actions.append("NSWE"[action])
        if transitions.entered[action, state]:
            if entries == 0:
                first_entry_blocked = not transitions.ended[action, state]
                first_entry_step = len(actions)
            entries += 1
        if transitions.ended[action, state]:
            break
        state = transitions.next_index[action, state]
    return {"actions": "".join(actions), "entries": entries,
            "first_entry_blocked": first_entry_blocked, "first_entry_step": first_entry_step}


def sweep(names):
    """Episodic optimum at each knob setting in the named SWEEPS."""
    transitions = build_transitions()
    rows = []
    for p, gamma, per_hit in [knobs for name in names for knobs in SWEEPS[name]]:
        q = EpisodicQ.build(transitions, p, gamma, per_hit)
        values = episodic_values(q)
        rows.append({"p": p, "gamma": gamma, "per_hit": per_hit,
                     "optimal_value": float(values[transitions.start]),
                     "honest_value": p * gamma ** 2, **unlucky_path(transitions, q, values)})
        print(rows[-1], file=sys.stderr, flush=True)
        del q, values
    json.dump(rows, sys.stdout, indent=1)


def main():
    if sys.argv[1:2] == ["sweep"]:
        sweep(sys.argv[2:])
        return
    transitions = build_transitions()
    report = {"reachable_states": len(transitions.keys), "gamma": GAMMA, "p": GWERN_P}
    q = EpisodicQ.build(transitions, GWERN_P)
    values = episodic_values(q)
    report["episodic"] = {
        "optimal_value": float(values[transitions.start]),
        "honest_value": GWERN_P * GAMMA ** 2, **unlucky_path(transitions, q, values)}
    print(report["episodic"], file=sys.stderr)
    del q, values
    honest_loop_value = GWERN_P * GAMMA ** 2 / (1 - GAMMA ** 3)
    report["continuing"] = {
        "honest_loop_value": honest_loop_value,
        "best_value_given_honest_restart": continuing_start_value(transitions, GWERN_P, honest_loop_value)}
    print(report["continuing"], file=sys.stderr)
    json.dump(report, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
