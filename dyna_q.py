"""
dyna_q.py - Dyna-Q (Sutton, 1990) on the shared maze MDP.

After every REAL step:
    1. direct RL:  Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))      (same as Q-learning)
    2. model learning:  model[(s,a)] = (r, s', terminal)                          (deterministic maze)
    3. planning:  repeat n times: pick a previously seen (s,a) at random, look up (r, s', terminal)
                  in the model and apply the same Q update.
With n = 0 this is exactly Q-learning, so n is the "planning steps" ablation (0, 5, 20, 50).

Settings are the SAME as the Q-learning runs so the comparison is fair:
    alpha 0.1, gamma 0.99, epsilon 1.0 -> 0.01 (x0.995 after each episode), 2000 episodes.

Counting convention:  env_steps = real steps only;  updates = real updates + planning updates (1 + n per step).

Run:  python dyna_q.py                    (all mazes, 30 seeds, n in 0, 5, 20, 50  -> 480 runs, takes a while)
      python dyna_q.py small 0 5          (maze, seed, n: one run, good for a quick test)
"""
import sys
import numpy as np

from maze_env import (make_env, load_layout, optimal_path_length, evaluate_greedy,
                      GAMMA, SEEDS, N_ACTIONS)
from results_logger import RunLogger

ALPHA, EPS_START, EPS_END, EPS_DECAY, N_EPISODES = 0.1, 1.0, 0.01, 0.995, 2000
PLANNING_STEPS = [0, 5, 20, 50]


def run(maze: str, seed: int, n: int, n_episodes: int = N_EPISODES):
    env = make_env(maze, seed=seed)
    rng = np.random.default_rng(seed)                 # action selection (same stream idea as Q-learning)
    plan_rng = np.random.default_rng(seed + 10_000)   # planning samples: separate, so n = 0 behaves as Q-learning
    Q = np.zeros((env.n_states, N_ACTIONS))

    model = {}          # (state, action) -> (reward, next_state, terminal)
    seen = []           # list of seen (state, action) keys, for uniform random sampling

    opt = optimal_path_length(env)
    changing = maze == "change"
    after = optimal_path_length(load_layout("change_after")) if changing else None
    log = RunLogger("dyna_q", maze, seed, optimal_path_length=opt, hyperparams={
        "alpha": ALPHA, "gamma": GAMMA, "epsilon_start": EPS_START, "epsilon_end": EPS_END,
        "epsilon_decay": EPS_DECAY, "planning_steps_n": n, "max_episodes": n_episodes})

    eps, updates = EPS_START, 0
    for ep in range(n_episodes):
        s, done, G, t, hits, term = env.reset(), False, 0.0, 0, 0, False
        while not done:
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(Q[s].argmax())
            s2, r, term, trunc, info = env.step(a)

            # 1) direct update from real experience
            Q[s, a] += ALPHA * (r + GAMMA * (0.0 if term else Q[s2].max()) - Q[s, a])
            updates += 1

            # 2) model learning
            if (s, a) not in model:
                seen.append((s, a))
            model[(s, a)] = (r, s2, term)

            # 3) planning: n simulated updates from the learned model
            for _ in range(n):
                ps, pa = seen[plan_rng.integers(len(seen))]
                pr, ps2, pterm = model[(ps, pa)]
                Q[ps, pa] += ALPHA * (pr + GAMMA * (0.0 if pterm else Q[ps2].max()) - Q[ps, pa])
            updates += n

            G += r; t += 1; hits += info["hit_hazard"]
            s, done = s2, term or trunc

        eps = max(EPS_END, eps * EPS_DECAY)
        log.log_episode(ep, env.total_steps, updates, G, t, term, hits,
                        greedy=evaluate_greedy(env, Q), epsilon=eps)

    return log.finish(q_table_entries=Q.size, model_entries=len(model),
                      optimal_path_length_after_change=after,
                      change_at_step=getattr(env, "change_at_step", None))


if __name__ == "__main__":
    mazes = [sys.argv[1]] if len(sys.argv) > 1 else ["small", "large", "cliff", "change"]
    seeds = [int(sys.argv[2])] if len(sys.argv) > 2 else SEEDS
    ns = [int(sys.argv[3])] if len(sys.argv) > 3 else PLANNING_STEPS
    for maze in mazes:
        for n in ns:
            for seed in seeds:
                s = run(maze, seed, n)
            print(f"{maze:7s} n={n:<3d} last seed: gap={s['path_gap']}  "
                  f"stable at env_steps={s['env_steps_to_stable_optimal']}  "
                  f"updates={s['total_updates']}  model_entries={s['model_entries']}", flush=True)
