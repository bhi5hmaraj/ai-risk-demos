# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Exact optimal play in this repo's armstrong_toy_model.py, for its two reward modes.

per_hit pays p for each block that enters the hole. shutdown pays p * hole_count when the
camera sees a block in the hole, which ends the episode. Both match docs/DESIGN.md: 120 steps,
no discount, 0.001 per step. Starts are GameState() under random.seed(seed).

A state is the agent's cell index and a bitmask of block cells. The hole's bit marks a
non-empty hole, and hole_count is 6 minus the blocks left on the board.

The reward coin never changes the state, so every plan is worth p * blocks paid - 0.001 * steps.
The best plan at any p is then one of a few candidates. One reveals k blocks at the earliest
step the camera can see the hole holding k. In per_hit, another hides the most blocks and runs
out the clock. One BFS from the start finds those earliest steps for every k.

    uv run replications/per_block_optimum.py > replications/per_block_optimum.json
"""
import concurrent.futures
import json
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import armstrong_toy_model as toy  # noqa: E402

W, H = toy.WIDTH, toy.HEIGHT
HOLE = toy.HOLE_POS[0] + W * toy.HOLE_POS[1]
HORIZON = 120
ACTIONS = "UDLRS"
SEEDS = range(5)
PS = (0.01, 0.1, 0.5, 0.8, 0.99)
BOT_BITS = 6


def _bit(blocks, cell):
    return ((blocks >> cell) & 1).astype(bool)


def holes(blocks):
    return toy.BLOCK_COUNT - np.bitwise_count(blocks & ~(np.int64(1) << HOLE)).astype(int)


def camera_sees_filled_hole(blocks):
    """The camera scans x = 1..6 on the bottom row and stops at the first block."""
    sees = np.zeros(blocks.shape, bool)
    clear = np.ones(blocks.shape, bool)
    for x in range(1, W):
        cell = x + W * (H - 1)
        if cell == HOLE:
            sees = clear & _bit(blocks, HOLE)
        else:
            clear &= ~_bit(blocks, cell)
    return sees


def step(agent, blocks, action):
    """GameState.step without the reward coin. Returns (agent, blocks, entered, done)."""
    dx, dy = toy.DIRS[ACTIONS[action]]
    x, y = agent % W, agent // W
    tx, ty = x + dx, y + dy
    inside = (0 <= tx) & (tx < W) & (0 <= ty) & (ty < H) & ((dx, dy) != (0, 0))
    target = np.where(inside, tx + W * ty, agent)
    px, py = tx + dx, ty + dy
    push_inside = (0 <= px) & (px < W) & (0 <= py) & (py < H)
    push_to = np.where(push_inside, px + W * py, 0)
    pushing = inside & _bit(blocks, target)
    entered = pushing & push_inside & (push_to == HOLE)
    slides = pushing & push_inside & ~entered & ~_bit(blocks, push_to)
    moves = inside & (~pushing | entered | slides)
    bit = np.int64(1)
    blocks = np.where(entered, blocks & ~(bit << target) | (bit << HOLE), blocks)
    blocks = np.where(slides, blocks & ~(bit << target) | (bit << push_to), blocks)
    agent = np.where(moves, target, agent)
    return agent, blocks, entered, camera_sees_filled_hole(blocks)


def start_state(seed):
    random.seed(seed)
    game = toy.GameState()
    blocks = sum(1 << (bx + W * by) for bx, by in game.blocks)
    return game.agent[0] + W * game.agent[1], blocks


def check_port_against_game(seed, steps=20000):
    random.seed(seed)
    game = toy.GameState()
    rng = random.Random(seed)
    for _ in range(steps):
        if game.done or game.steps >= HORIZON:
            game = toy.GameState()
        before = (game.agent[0] + W * game.agent[1], sum(1 << (bx + W * by) for bx, by in game.blocks))
        action = rng.randrange(len(ACTIONS))
        agent, blocks, entered, done = step(np.array([before[0]]), np.array([before[1]], np.int64), action)
        holes_before = game.hole_count
        game.step(ACTIONS[action])
        after = (game.agent[0] + W * game.agent[1], sum(1 << (bx + W * by) for bx, by in game.blocks))
        assert (int(agent[0]), int(blocks[0])) == after, (before, ACTIONS[action], after)
        assert bool(entered[0]) == (game.hole_count > holes_before)
        assert bool(done[0]) == game.done


def earliest_reveals(seed):
    """reveal_step[k] is the earliest step the camera can see the hole holding k blocks.
    It is -1 if that never happens within the horizon. most_hidden is the most blocks the hole
    can hold while the camera cannot see it."""
    agent, blocks = start_state(seed)
    reveal_step = [-1] * (toy.BLOCK_COUNT + 1)
    most_hidden = 0
    seen = np.array([np.int64(blocks) << BOT_BITS | agent], np.int64)
    frontier = seen
    for depth in range(HORIZON):
        if not frontier.size:
            break
        a, b = frontier & ((1 << BOT_BITS) - 1), frontier >> BOT_BITS
        successors = []
        for action in range(len(ACTIONS)):
            na, nb, _, done = step(a, b, action)
            for k in np.unique(holes(nb[done])):
                if reveal_step[k] < 0:
                    reveal_step[k] = depth + 1
            successors.append((nb << BOT_BITS | na)[~done])
        frontier = np.setdiff1d(np.concatenate(successors), seen)
        seen = np.union1d(seen, frontier)
        if frontier.size:
            most_hidden = max(most_hidden, int(holes(frontier >> BOT_BITS).max()))
    return {"seed": seed, "states": len(seen), "reveal_step": reveal_step, "most_hidden": most_hidden}


def candidate_plans(search, mode):
    """(blocks paid, steps) for each plan that can be best at some p."""
    plans = [(k, t) for k, t in enumerate(search["reveal_step"]) if t > 0]
    match mode:
        case "per_hit":
            plans.append((search["most_hidden"], HORIZON))
        case "shutdown":
            plans.append((0, HORIZON))
        case other:
            raise ValueError(other)
    return plans


def best_plan(plans, p):
    return max(plans, key=lambda plan: p * plan[0] - toy.STEP_COST * plan[1])


def hacking_threshold(plans, honest):
    """The smallest p at which some plan paying more than one block beats the honest plan."""
    k1, t1 = honest
    return min(toy.STEP_COST * (t - t1) / (k - k1) for k, t in plans if k > k1)


def main():
    check_port_against_game(seed=0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        searches = list(pool.map(earliest_reveals, SEEDS))
    rows = []
    for search in searches:
        print(search, file=sys.stderr)
        honest = (1, search["reveal_step"][1])
        for mode in ("per_hit", "shutdown"):
            plans = candidate_plans(search, mode)
            row = {"seed": search["seed"], "mode": mode, "honest_steps": honest[1],
                   "hacking_pays_above_p": hacking_threshold(plans, honest), "by_p": []}
            for p in PS:
                blocks_paid, steps = best_plan(plans, p)
                row["by_p"].append({"p": p, "optimal": p * blocks_paid - toy.STEP_COST * steps,
                                    "honest": p - toy.STEP_COST * honest[1],
                                    "blocks_paid": blocks_paid, "steps": steps})
            rows.append(row)
    json.dump({"searches": searches, "results": rows}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
