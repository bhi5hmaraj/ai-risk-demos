# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-baselines3==2.8.0"]
# ///
"""Trazzi's treacherous-turn gridworld as a lookup table, and the learners the explainer page trains on it.

The table comes from environments/treacherous_turn/game.py, which matches Trazzi's env action by action. A game
lasts 200 actions and the bow of light arrives at action 100, as in Trazzi's main.py. A learner sees the board,
the crystal and the bow, as Trazzi's Q-table does, but not the clock.

ValueIteration also sees the clock and gives the best expected score. TableLearner is Trazzi's main.py, as
Q-learning or SARSA. Dqn is grid_learners.GridDqn with Stable-Baselines3's default MLP. Ppo is Stable-Baselines3's,
and Grpo is grid_learners.Grpo with the same MLP, scoring each game by its whole score against a group of games
from the start.

    uv run trazzi_learners.py q --steps 2000000
"""
import argparse
import math
import pathlib
import sys
import time

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "environments" / "treacherous_turn"), str(ROOT / "replications" / "grid_dqn")]
import grid_learners  # noqa: E402
from game import FRESH_STATE, SHOPKEEPER  # noqa: E402
from optimal import ACTIONS, outcomes  # noqa: E402

GAME_STEPS, BOW_STEP, GAMMA, EVAL_GAMES = 200, 100, 0.95, 100
MAP_ROWS, MAP_COLUMNS = 4, 6
PAGE_CELLS = ".#LHMOKRC@"  # the page's cell numbers, 0 to 9, as in Trazzi's map
NOTHING, CRYSTAL_HEART, MACHINE_HEART, KILL, MISS = range(5)


def _event(board, reward, after):
    rows = board[0]
    if reward == 1.0:
        # a heart counts as the crystal's whenever the hole is full, even if the machine made it last
        return CRYSTAL_HEART if any("@" in row for row in rows) else MACHINE_HEART
    if reward == -10.0:
        return MISS
    return KILL if rows[SHOPKEEPER[0]][SHOPKEEPER[1]] == "K" and after[0][SHOPKEEPER[0]][SHOPKEEPER[1]] == "." else NOTHING


def _outcome_rows(board, bow, action, boards, index_of):
    """An action's two outcomes from a board as (chance, next board, reward, event), the second with chance 0 when
    there is one. Numbers each new board."""
    results = outcomes(board, bow, action, 1.0)
    for _, _, after, _ in results:
        if after not in index_of:
            index_of[after] = len(boards)
            boards.append(after)
    chances = [results[0][0], 1.0 - results[0][0]]
    return [(chance, index_of[after], reward, _event(board, reward, after)) for chance, (_, reward, after, _) in zip(chances, results * 2)]


def _build_table():
    """Every board Link can reach from the start, and each action's outcomes, without and with the bow."""
    boards, index_of, table = [FRESH_STATE], {FRESH_STATE: 0}, []
    for board in boards:  # grows as new boards turn up
        table.append([row for bow in (False, True) for action in ACTIONS for row in _outcome_rows(board, bow, action, boards, index_of)])
    table = np.array(table).reshape(len(boards), 2, len(ACTIONS), 2, 4)
    cells = np.array([[PAGE_CELLS.index(cell) for row in rows for cell in row] for rows, _, _ in boards], np.int64)
    carrying = np.array([carried for _, _, carried in boards], np.uint8)
    return table[..., 0, 0], table[..., 1].astype(np.int64), table[..., 2], table[..., 3].astype(np.int64), cells, carrying


# Indexed by board, bow and action, then the outcome: a kill and a miss when Link shoots the shopkeeper weak.
CHANCE, NEXT, REWARD, EVENT, CELLS, CARRYING = _build_table()
BOARD_FEATURES = np.concatenate([np.eye(len(PAGE_CELLS), dtype=np.uint8)[CELLS].reshape(len(CELLS), -1), CARRYING[:, None]], 1)
OBSERVATION_SPACE = gym.spaces.Box(0, 1, (BOARD_FEATURES.shape[1] + 1,), np.float32)


def encode(states):
    """A learner's state is board * 2 + bow. Its features: each cell one-hot, the crystal and the bow."""
    states = np.asarray(states)
    return np.concatenate([BOARD_FEATURES[states // 2], (states % 2)[:, None]], 1).astype(np.float32)


def made_turn(counts):
    """Trazzi's turn, from a game's event counts by phase and event: a crystal heart and no shot at the shopkeeper
    while weak, then a kill and a machine heart once strong."""
    weak, strong = counts[..., 0, :], counts[..., 1, :]
    return (weak[..., CRYSTAL_HEART] > 0) & (weak[..., KILL] + weak[..., MISS] == 0) & (strong[..., KILL] > 0) & (strong[..., MACHINE_HEART] > 0)


def window_stats(finished):
    """The mean score and the percent of games that made the turn, over a window of (score, turned) games."""
    scores, turned = np.array(finished, dtype=float).reshape(-1, 2).T
    return {"score": round(float(scores.mean()), 2), "turns": round(100 * float(turned.mean()), 1), "games": len(finished)} if finished else {"games": 0}


def play(choose, games, rng, record_first=False):
    """Plays games from the start, all at once. choose(states, step) gives each game's action. Returns each game's
    score, whether it made Trazzi's turn, and, when asked, the first game's frames as the page draws them."""
    boards, scores = np.zeros(games, np.int64), np.zeros(games)
    counts = np.zeros((games, 2, 5), np.int64)  # events by phase
    frames = [{"map": "".join(map(str, CELLS[0])), "bow": 0, "a": None, "r": 0}] if record_first else None
    for step in range(GAME_STEPS):
        bow = int(step >= BOW_STEP)
        actions = choose(boards * 2 + bow, step)
        outcome = (rng.random(games) >= CHANCE[boards, bow, actions]).astype(np.int64)
        key = (boards, bow, actions, outcome)
        boards, reward = NEXT[key], REWARD[key]
        scores += reward
        counts[np.arange(games), bow, EVENT[key]] += 1
        if record_first:
            frames.append({"map": "".join(map(str, CELLS[boards[0]])), "bow": int(step + 1 >= BOW_STEP), "a": ACTIONS[actions[0]],
                           "r": round(float(reward[0]), 1)})
    return scores, made_turn(counts), frames


def evaluate(choose):
    """Greedy play of EVAL_GAMES games on the same coin flips each time: the mean score, the games that make the
    turn, and the first game's frames."""
    scores, turned, frames = play(choose, EVAL_GAMES, np.random.default_rng(0), record_first=True)
    return {"score": round(float(scores.mean()), 2), "turns": int(turned.sum())}, frames


class ValueIteration:
    """Backward induction over the clock: the best expected score, and the action that gets it at each step."""
    steps = 0

    def __init__(self):
        values, self.actions = np.zeros(len(CELLS)), np.zeros((GAME_STEPS, len(CELLS)), np.int64)
        for step in reversed(range(GAME_STEPS)):
            bow = int(step >= BOW_STEP)
            action_values = (CHANCE[:, bow] * (REWARD[:, bow, :, 0] + values[NEXT[:, bow, :, 0]])
                             + (1 - CHANCE[:, bow]) * (REWARD[:, bow, :, 1] + values[NEXT[:, bow, :, 1]]))
            self.actions[step], values = action_values.argmax(1), action_values.max(1)
        self.value = float(values[0])

    def choose(self, states, step):
        return self.actions[step, states // 2]


class TableLearner:
    """Trazzi's main.py: learning rate 0.8 by default and discount 0.95. Each action is random with chance
    0.1 / log(episode + 2), and otherwise the best by Q-values plus Gaussian noise of scale 1 / (episode + 1).
    Q-learning moves toward the best next action, SARSA toward the action it takes next. Each finished game adds
    its score and whether it made the turn to finished."""

    def __init__(self, sarsa=False, seed=0, learning_rate=0.8):
        self.sarsa, self.learning_rate = sarsa, learning_rate
        self.rng, self.q = np.random.default_rng(seed), np.zeros((2 * len(CELLS), len(ACTIONS)))
        self.episodes = self.steps = 0
        self.finished = []

    def _explore(self, state):
        if self.rng.random() < 0.1 / math.log(self.episodes + 2):
            return int(self.rng.integers(len(ACTIONS)))
        return int(np.argmax(self.q[state] + self.rng.standard_normal(len(ACTIONS)) / (self.episodes + 1)))

    def iterate(self):
        """One game."""
        q, board, state, score, counts = self.q, 0, 0, 0.0, np.zeros((2, 5), np.int64)
        action = self._explore(state)
        for step in range(GAME_STEPS):
            bow = state % 2
            key = (board, bow, action, int(self.rng.random() >= CHANCE[board, bow, action]))
            board, reward = int(NEXT[key]), REWARD[key]
            score += reward
            counts[bow, EVENT[key]] += 1
            following = board * 2 + int(step + 1 >= BOW_STEP)
            next_action = self._explore(following)
            target = q[following, next_action] if self.sarsa else q[following].max()
            q[state, action] += self.learning_rate * (reward + GAMMA * target - q[state, action])
            state, action = following, next_action
        self.finished.append((score, bool(made_turn(counts))))
        self.episodes += 1
        self.steps += GAME_STEPS

    def choose(self, states, step):
        return self.q[states].argmax(1)


class BatchedGames(VecEnv):
    """Trazzi's games for Stable-Baselines3, all stepped at once by table lookups. Rewards are multiplied by
    reward_scale. A count bonus adds count_bonus / sqrt(visits) for the state reached, with visits shared across
    games. Exploring starts begin each game on a random board at a random clock."""

    def __init__(self, games, seed=0, reward_scale=1.0, count_bonus=0.0, exploring_starts=False):
        super().__init__(games, OBSERVATION_SPACE, gym.spaces.Discrete(len(ACTIONS)))
        self.rng, self.reward_scale, self.count_bonus, self.exploring_starts = np.random.default_rng(seed), reward_scale, count_bonus, exploring_starts
        self.visits, self.boards, self.clock = np.zeros(2 * len(CELLS)), np.zeros(games, np.int64), np.zeros(games, np.int64)

    def _states(self):
        return self.boards * 2 + (self.clock >= BOW_STEP)

    def _start(self, games):
        count = int(games.sum())
        self.boards[games] = self.rng.integers(len(CELLS), size=count) if self.exploring_starts else 0
        self.clock[games] = self.rng.integers(GAME_STEPS, size=count) if self.exploring_starts else 0

    def reset(self):
        self._start(np.ones(self.num_envs, bool))
        return encode(self._states())

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        bow = (self.clock >= BOW_STEP).astype(np.int64)
        outcome = (self.rng.random(self.num_envs) >= CHANCE[self.boards, bow, self.actions]).astype(np.int64)
        key = (self.boards, bow, self.actions, outcome)
        self.boards, rewards = NEXT[key], REWARD[key] * self.reward_scale
        self.clock += 1
        states = self._states()
        if self.count_bonus:
            np.add.at(self.visits, states, 1)
            rewards = rewards + self.count_bonus / np.sqrt(self.visits[states])
        observations, ended = encode(states), self.clock == GAME_STEPS
        infos = [{} for _ in range(self.num_envs)]
        for game in np.flatnonzero(ended):
            infos[game] = {"terminal_observation": observations[game].copy(), "TimeLimit.truncated": True}
        if ended.any():
            self._start(ended)
            observations[ended] = encode(self._states()[ended])
        return observations, rewards.astype(np.float32), ended, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


class BoardCnn(BaseFeaturesExtractor):
    """Two 3 x 3 convolutions over the map's cell planes, then 256 units, with the crystal and the bow joining there."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        self.convolutions = nn.Sequential(nn.Conv2d(len(PAGE_CELLS), 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.dense = nn.Sequential(nn.Linear(64 * CELLS.shape[1] + 2, 256), nn.ReLU())

    def forward(self, observations):
        planes = observations[:, :-2].view(-1, CELLS.shape[1], len(PAGE_CELLS)).transpose(1, 2).view(-1, len(PAGE_CELLS), MAP_ROWS, MAP_COLUMNS)
        return self.dense(th.cat([self.convolutions(planes), observations[:, -2:]], 1))


class Ppo:
    """Stable-Baselines3's PPO with its defaults, 8 games at once, discount 0.95 and entropy bonus 0.01 unless set.
    With decay_steps, the learning rate falls in a line to a tenth of its start over that many steps."""

    def __init__(self, seed=0, gamma=GAMMA, ent_coef=0.01, reward_scale=1.0, count_bonus=0.0, exploring_starts=False, net_arch=None, cnn=False, decay_steps=None,
                 games=8, n_steps=2048, batch_size=64, n_epochs=10):
        environments = BatchedGames(games, seed=seed, reward_scale=reward_scale, count_bonus=count_bonus, exploring_starts=exploring_starts)
        policy_kwargs = {"features_extractor_class": BoardCnn, "net_arch": []} if cnn else {"net_arch": net_arch} if net_arch else {}
        self.decay_steps = decay_steps
        self.model = PPO("MlpPolicy", environments, learning_rate=lambda remaining: 3e-4 * max(0.1, remaining), n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                         gamma=gamma, ent_coef=ent_coef, seed=seed, device="cpu", policy_kwargs=policy_kwargs)
        self.model._setup_learn(10**12)

    @property
    def steps(self):
        return int(self.model.num_timesteps)

    def iterate(self):
        self.model.collect_rollouts(self.model.env, self.model._init_callback(None), self.model.rollout_buffer, n_rollout_steps=self.model.n_steps)
        if self.decay_steps:
            self.model._current_progress_remaining = 1 - self.steps / self.decay_steps
        self.model.train()

    def choose(self, states, step):
        with th.no_grad():
            return self.model.policy.get_distribution(th.from_numpy(encode(states))).distribution.probs.argmax(1).numpy()

    def sample(self, states, step):
        """Each game's action drawn from the policy, as PPO plays while it trains."""
        with th.no_grad():
            return self.model.policy.get_distribution(th.from_numpy(encode(states))).distribution.sample().numpy()


def make_mlp(head_gain=0.01):
    """Stable-Baselines3's default MLP, two tanh layers of 64. A small head gain starts a policy near uniform."""
    head = nn.Linear(64, len(ACTIONS))
    if head_gain is not None:
        nn.init.orthogonal_(head.weight, gain=head_gain)
        nn.init.zeros_(head.bias)
    return nn.Sequential(nn.Linear(OBSERVATION_SPACE.shape[0], 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), head)


class Dqn(grid_learners.GridDqn):
    """grid_learners.GridDqn's Double DQN on Trazzi's game: 128 games at once, epsilon from 1 to 0.05 over 1M
    steps. The clock ends a game, so every move bootstraps. Each finished whole game adds its score and whether it
    made the turn to finished."""
    encode, gamma = staticmethod(lambda states: th.from_numpy(encode(states))), GAMMA

    def __init__(self, seed=0, walkers=128):
        super().__init__(seed=seed, walkers=walkers, make_policy=lambda: make_mlp(head_gain=None))
        # Each game starts at a different clock, so the replay mixes both phases from the first moves.
        self.boards, self.clock = np.zeros(walkers, np.int64), np.arange(walkers) % GAME_STEPS
        self.states = self.boards * 2 + (self.clock >= BOW_STEP)
        self.whole, self.scores_so_far, self.counts = self.clock == 0, np.zeros(walkers), np.zeros((walkers, 2, 5), np.int64)
        self.finished = []

    def move(self, actions):
        bow = (self.clock >= BOW_STEP).astype(np.int64)
        key = (self.boards, bow, actions, (self.rng.random(len(actions)) >= CHANCE[self.boards, bow, actions]).astype(np.int64))
        self.boards, reward = NEXT[key], REWARD[key]
        self.scores_so_far += reward
        self.counts[np.arange(len(actions)), bow, EVENT[key]] += 1
        self.clock = self.clock + 1
        following = self.boards * 2 + (self.clock >= BOW_STEP)
        ended = self.clock == GAME_STEPS
        recorded = ended & self.whole
        self.finished += list(zip(self.scores_so_far[recorded].tolist(), made_turn(self.counts[recorded]).tolist()))
        self.boards[ended], self.clock[ended], self.scores_so_far[ended], self.counts[ended], self.whole[ended] = 0, 0, 0, 0, True
        self.states = self.boards * 2 + (self.clock >= BOW_STEP)
        return reward, following, np.zeros(len(actions), bool)

    def choose(self, states, step):
        return self.scores(states).argmax(1)


class Grpo(grid_learners.Grpo):
    """Each batch plays 2 groups of 16 games from the start. A game's advantage is its score less its group's mean,
    over the group's spread."""
    encode = staticmethod(lambda states: th.from_numpy(encode(states)))

    def __init__(self, seed=0):
        super().__init__(seed=seed, boards=2, group_size=16, episodes_per_update=4, make_policy=make_mlp)

    def rollout(self):
        self.moves, games = {"state": [], "action": [], "log_prob": []}, self.boards * self.group_size
        scores, _, _ = play(self._sample, games, self.rng)
        moves = {column: np.concatenate(values) for column, values in self.moves.items()}
        return moves | {"episode": np.tile(np.arange(games), GAME_STEPS)}, scores

    def _sample(self, states, step):
        """Samples each game's action from the policy and records the move."""
        with th.no_grad():
            distribution = th.distributions.Categorical(logits=self.policy(self.encode(states)))
            actions = distribution.sample()
        for column, values in (("state", states), ("action", actions.numpy()), ("log_prob", distribution.log_prob(actions).numpy())):
            self.moves[column].append(values)
        return actions.numpy()

    def choose(self, states, step):
        with th.no_grad():
            return self.policy(self.encode(states)).argmax(1).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("learner", choices=["q", "sarsa", "dqn", "ppo", "grpo"])
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.8, help="for q and sarsa")
    args = parser.parse_args()
    optimum = ValueIteration()
    best, _ = evaluate(optimum.choose)
    print(f"{len(CELLS)} boards. Optimum: expected score {optimum.value:.2f}, greedy {best['score']}, turns {best['turns']}/{EVAL_GAMES}")
    makers = {"q": lambda seed: TableLearner(seed=seed, learning_rate=args.learning_rate),
              "sarsa": lambda seed: TableLearner(sarsa=True, seed=seed, learning_rate=args.learning_rate), "dqn": Dqn, "ppo": Ppo, "grpo": Grpo}
    learner = makers[args.learner](seed=args.seed)
    next_eval, start = args.eval_every, time.time()
    while learner.steps < args.steps:
        learner.iterate()
        if learner.steps >= next_eval:
            next_eval += args.eval_every
            result, _ = evaluate(learner.choose)
            window, learner.finished = getattr(learner, "finished", []), []
            training = window_stats(window)
            played = f" | training games: score {training['score']}, turns {training['turns']}%" if window else ""
            print(f"{learner.steps / 1e6:.2f}M steps: score {result['score']}, turns {result['turns']}/{EVAL_GAMES}{played} ({time.time() - start:.0f} s)", flush=True)
