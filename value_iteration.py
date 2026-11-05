"""
value_iteration.py - Value Iteration on the shared maze MDP.

Bellman optimality update, applied to every state each sweep:
    V(s) <- max_a  sum_{s'} P(s'|s,a) * [ r + gamma * V(s') * (1 - done) ]
Stops when the largest change in a sweep (delta) falls below theta.
Then the greedy policy is  pi(s) = argmax_a Q(s,a)  with the final V.

Run:  python value_iteration.py            (all mazes, all 30 seeds)
      python value_iteration.py small 1    (one maze, one seed)

Counting convention (agree with the group!): one "update" = one Q(s,a) backup, so a sweep
costs (number of non-wall states) x 4 updates. This is comparable to one Q-learning update,
which also changes one Q(s,a) entry. env_steps is 0 because VI never interacts with the env.
"""
import sys
import numpy as np

from maze_env import (MazeEnv, make_env, load_layout, optimal_path_length, evaluate_greedy,
                      GAMMA, SEEDS, N_ACTIONS)
from results_logger import RunLogger

THETA = 1e-6
MAX_SWEEPS = 10_000


def q_from_v(env: MazeEnv, V: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Q(s,a) = expected one-step return using the model and the current V."""
    P = env.transition_model()
    Q = np.zeros((env.n_states, N_ACTIONS))
    for s in env.valid_states:
        if s == env.goal:
            continue                      # terminal: Q stays 0
        for a in range(N_ACTIONS):
            Q[s, a] = sum(p * (r + gamma * V[s2] * (not done)) for p, s2, r, done in P[s][a])
    return Q


def value_iteration(env: MazeEnv, gamma: float = GAMMA, theta: float = THETA, on_sweep=None):
    """Returns (V, Q, sweeps, final_delta). on_sweep(sweep, V, Q, updates) is called after every sweep."""
    V = np.zeros(env.n_states)
    states = [s for s in env.valid_states if s != env.goal]
    updates_per_sweep = len(env.valid_states) * N_ACTIONS
    updates = 0
    for sweep in range(1, MAX_SWEEPS + 1):
        Q = q_from_v(env, V, gamma)
        V_new = V.copy()
        for s in states:
            V_new[s] = Q[s].max()
        delta = float(np.abs(V_new - V).max())
        V = V_new
        updates += updates_per_sweep
        if on_sweep:
            on_sweep(sweep, V, q_from_v(env, V, gamma), updates)
        if delta < theta:
            return V, q_from_v(env, V, gamma), sweep, delta
    return V, q_from_v(env, V, gamma), MAX_SWEEPS, delta


def run(maze: str, seed: int):
    if maze == "change":
        # VI knows the model, so it simply re-solves each layout. Log the BEFORE layout as the main run
        # and report the re-planning cost after the change in `notes`.
        env = MazeEnv(load_layout("change_before"), seed=seed)
        env_after = MazeEnv(load_layout("change_after"), seed=seed)
    else:
        env, env_after = make_env(maze, seed=seed), None

    opt = optimal_path_length(env)
    log = RunLogger("value_iteration", maze, seed, optimal_path_length=opt,
                    hyperparams={"gamma": GAMMA, "theta": THETA})

    def on_sweep(sweep, V, Q, updates):
        log.log_episode(sweep, env_steps=0, updates=updates, greedy=evaluate_greedy(env, Q))

    V, Q, sweeps, delta = value_iteration(env, on_sweep=on_sweep)

    notes = ""
    if env_after is not None:
        _, Q2, sweeps2, _ = value_iteration(env_after)
        gap2 = evaluate_greedy(env_after, Q2)["path_length"]
        notes = f"replan_sweeps_after_change={sweeps2}; path_after_change={gap2}"
    return log.finish(sweeps_to_converge=sweeps, final_delta=delta, q_table_entries=Q.size,
                      model_entries="", notes=notes), V, Q


if __name__ == "__main__":
    mazes = [sys.argv[1]] if len(sys.argv) > 1 else ["small", "large", "cliff", "change"]
    seeds = [int(sys.argv[2])] if len(sys.argv) > 2 else SEEDS
    for maze in mazes:
        for seed in seeds:
            summary, V, Q = run(maze, seed)
        print(f"{maze:7s} sweeps={summary['sweeps_to_converge']:3d}  "
              f"path={summary['final_path_length']} (optimal {summary['optimal_path_length']})  "
              f"gap={summary['path_gap']}  updates={summary['total_updates']}  {summary['notes']}")
