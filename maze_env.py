"""
maze_env.py - the ONE shared maze MDP for the whole group.

Do not edit without telling everyone. All methods (Value Iteration, Q-learning,
Dyna-Q) must import this file so results are comparable.

Quick start
-----------
    from maze_env import make_env, optimal_path_length, evaluate_greedy, SEEDS, GAMMA

    env = make_env("small", seed=0)
    s = env.reset()
    s, r, terminated, truncated, info = env.step(3)   # 3 = RIGHT

Model-free methods (Q-learning, Dyna-Q): use reset() / step().
Model-based methods (Value Iteration): use env.transition_model(), which returns
    P[s][a] = list of (prob, next_state, reward, done)
Terminal (goal) states are absorbing: treat V(next_state) as 0 when done is True.

States are numbered row * n_cols + col over the WHOLE grid (wall cells included but
never visited), so the Q-table shape (n_states, 4) is identical before and after the
maze changes.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

#to make sure it is compatible with the other files, we define some constants here
GAMMA = 0.99             # discount factor, same for every method
STEP_REWARD = -1.0       # every move (including bumping into a wall)
GOAL_REWARD = 100.0      # entering the goal; episode terminates
HAZARD_REWARD = -100.0   # entering a hazard 'H'; agent is sent back to start (episode continues)
MAX_STEPS = 500          # per-episode time limit (episode is truncated after this)
SEEDS = list(range(30))  # every method must use these seeds
CHANGE_AT_STEP = 20_000  # changing-maze test: layout switches after this many real env steps

N_ACTIONS = 4
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_ARROWS = ["^", "v", "<", ">"]
MAZE_DIR = Path(__file__).parent / "mazes"


def load_layout(name: str) -> str:
    return (MAZE_DIR / f"{name}.txt").read_text()


class MazeEnv:
    """Grid maze. Legend: '#' wall, '.' free, 'S' start, 'G' goal, 'H' hazard."""

    def __init__(self, layout: str, slip_prob: float = 0.0,
                 max_steps: int = MAX_STEPS, seed: int = 0):
        self.slip_prob = slip_prob   # prob. the action is replaced by a random one (default: deterministic)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.total_steps = 0         # real env steps over all episodes (never reset)
        self._set_layout(layout)
        self.state = self.start
        self.t = 0

    # Building a maze from a text layout is slow, so we provide a classmethod to load from a file.
    @classmethod
    def from_name(cls, name: str, **kw) -> "MazeEnv":
        return cls(load_layout(name), **kw)

    def _set_layout(self, layout: str) -> None:
        rows = [line.rstrip() for line in layout.splitlines() if line.strip()]
        if len({len(r) for r in rows}) != 1:
            raise ValueError("All maze rows must have the same width")
        if hasattr(self, "n_rows") and (len(rows), len(rows[0])) != (self.n_rows, self.n_cols):
            raise ValueError("A changed maze must keep the same dimensions")
        self.layout_text = "\n".join(rows)
        self.n_rows, self.n_cols = len(rows), len(rows[0])
        self.n_states, self.n_actions = self.n_rows * self.n_cols, N_ACTIONS
        self.walls, self.hazards = set(), set()
        start = goal = None
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                s = r * self.n_cols + c
                if ch == "#":
                    self.walls.add(s)
                elif ch == "H":
                    self.hazards.add(s)
                elif ch == "S":
                    start = s
                elif ch == "G":
                    goal = s
                elif ch != ".":
                    raise ValueError(f"Unknown maze character {ch!r}")
        if start is None or goal is None:
            raise ValueError("Maze needs exactly one 'S' and one 'G'")
        self.start, self.goal = start, goal
        self._P = None  # invalidate cached transition model

    # --------- deterministic dynamics ----------
    def _move(self, s: int, a: int):
        """Deterministic move. Returns (next_state, reward, terminal, hit_hazard)."""
        r, c = divmod(s, self.n_cols)
        dr, dc = ACTION_DELTAS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols and (nr * self.n_cols + nc) not in self.walls:
            ns = nr * self.n_cols + nc
        else:
            ns = s  # bumped into a wall / boundary
        if ns == self.goal:
            return ns, GOAL_REWARD, True, False
        if ns in self.hazards:
            return self.start, HAZARD_REWARD, False, True
        return ns, STEP_REWARD, False, False

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state, self.t = self.start, 0
        return self.state

    def step(self, action: int):
        """Returns (next_state, reward, terminated, truncated, info)."""
        if self.slip_prob > 0 and self.rng.random() < self.slip_prob:
            action = int(self.rng.integers(N_ACTIONS))
        ns, reward, terminated, hit = self._move(self.state, int(action))
        self.state = ns
        self.t += 1
        self.total_steps += 1
        truncated = (self.t >= self.max_steps) and not terminated
        return ns, reward, terminated, truncated, {"hit_hazard": hit}

    # ---------- model access for dynamic programming ----------
    def transition_model(self):
        """P[s][a] = [(prob, next_state, reward, done), ...] for the CURRENT layout."""
        if self._P is None:
            P = {}
            for s in range(self.n_states):
                P[s] = {}
                for a in range(N_ACTIONS):
                    if s in self.walls:
                        P[s][a] = [(1.0, s, 0.0, False)]      # unreachable
                    elif s == self.goal:
                        P[s][a] = [(1.0, s, 0.0, True)]       # absorbing terminal
                    else:
                        outcomes = []
                        for a2 in range(N_ACTIONS):
                            p = self.slip_prob / N_ACTIONS + ((1 - self.slip_prob) if a2 == a else 0.0)
                            if p > 0:
                                ns, r, done, _ = self._move(s, a2)
                                outcomes.append((p, ns, r, done))
                        P[s][a] = outcomes
            self._P = P
        return self._P

    @property
    def valid_states(self):
        return [s for s in range(self.n_states) if s not in self.walls]

    # ---------- display ----------
    def render(self, policy=None) -> str:
        """Text picture. Pass a Q-table (n_states,4) or policy array (n_states,) to draw arrows."""
        pi = _as_policy(policy) if policy is not None else None
        lines = []
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                s = r * self.n_cols + c
                if s in self.walls:
                    row += "#"
                elif s == self.goal:
                    row += "G"
                elif s in self.hazards:
                    row += "H"
                elif pi is not None:
                    row += ACTION_ARROWS[pi[s]]
                else:
                    row += "S" if s == self.start else "."
            lines.append(row)
        return "\n".join(lines)


class ChangingMazeEnv(MazeEnv):
    """Switches from layout `before` to `after` once total real steps reach `change_at_step`.

    The agent is NOT told about the change. Use `env.changed` / `env.change_step` to log it.
    """

    def __init__(self, before: str, after: str, change_at_step: int = CHANGE_AT_STEP, **kw):
        super().__init__(before, **kw)
        self._after = after
        self.change_at_step = change_at_step
        self.changed = False
        self.change_step = None

    def step(self, action: int):
        if not self.changed and self.total_steps >= self.change_at_step:
            self._set_layout(self._after)
            self.changed, self.change_step = True, self.total_steps
            if self.state in self.walls:   # a wall appeared on the agent's cell
                self.state = self.start
        return super().step(action)


def make_env(name: str, seed: int = 0, **kw) -> MazeEnv:
    """name in {'small', 'large', 'cliff', 'change'}."""
    if name == "change":
        return ChangingMazeEnv(load_layout("change_before"), load_layout("change_after"), seed=seed, **kw)
    return MazeEnv.from_name(name, seed=seed, **kw)


# =====================  shared benchmark + evaluation  =====================
def optimal_path_length(env_or_layout) -> int | None:
    """Fewest steps from S to G avoiding walls and hazards (BFS ground truth).

    Matches the optimal policy of the contract MDP when slip_prob = 0.
    For the changing maze, call it once per layout (before and after).
    """
    env = env_or_layout if isinstance(env_or_layout, MazeEnv) else MazeEnv(env_or_layout)
    seen, q = {env.start: 0}, deque([env.start])
    while q:
        s = q.popleft()
        if s == env.goal:
            return seen[s]
        r, c = divmod(s, env.n_cols)
        for dr, dc in ACTION_DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.n_rows and 0 <= nc < env.n_cols:
                ns = nr * env.n_cols + nc
                if ns not in env.walls and ns not in env.hazards and ns not in seen:
                    seen[ns] = seen[s] + 1
                    q.append(ns)
    return None


def _as_policy(policy) -> np.ndarray:
    policy = np.asarray(policy)
    return policy.argmax(axis=1) if policy.ndim == 2 else policy.astype(int)


def evaluate_greedy(env: MazeEnv, policy, max_steps: int | None = None) -> dict:
    """Run the greedy policy once on a deterministic copy of the env's CURRENT layout.

    `policy` is a Q-table (n_states, 4), a V-based greedy policy array (n_states,) of actions.
    Does not touch the training env (step counters, rng, state). Ties in argmax -> lowest action index.
    Returns dict(reached_goal, path_length, total_return, hazard_hits). path_length is None if the
    goal was not reached within max_steps.
    """
    pi = _as_policy(policy)
    test = MazeEnv(env.layout_text, slip_prob=0.0, max_steps=max_steps or env.max_steps)
    s, total, hits, steps = test.reset(), 0.0, 0, 0
    while True:
        s, r, terminated, truncated, info = test.step(pi[s])
        total += r
        steps += 1
        hits += info["hit_hazard"]
        if terminated or truncated:
            break
    return {"reached_goal": bool(terminated), "path_length": steps if terminated else None,
            "total_return": total, "hazard_hits": hits}


if __name__ == "__main__":
    for name in ["small", "large", "cliff", "change"]:
        e = make_env(name)
        print(f"{name:7s} {e.n_rows}x{e.n_cols}  optimal path = {optimal_path_length(e)}")
    e = make_env("change")
    e._set_layout(load_layout("change_after"))
    print(f"change (after) optimal path = {optimal_path_length(e)}")
    print(make_env("cliff").render())
