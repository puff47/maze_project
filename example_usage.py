"""
example_usage.py - shows how to plug a method into the shared env + logger.
This is a minimal tabular Q-learning demo, NOT the official Q-learning implementation:
copy the structure (env, seeds, evaluate_greedy, RunLogger), replace the learning code with yours.
"""
import numpy as np
from maze_env import make_env, optimal_path_length, evaluate_greedy, SEEDS, GAMMA, N_ACTIONS, load_layout, MazeEnv
from results_logger import RunLogger

def run_q_learning(maze, seed, n_episodes=300, alpha=0.1, eps_start=1.0, eps_end=0.05, eps_decay=0.98):
    env = make_env(maze, seed=seed)
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, N_ACTIONS))
    opt = optimal_path_length(env)
    after = optimal_path_length(load_layout("change_after")) if maze == "change" else None
    log = RunLogger("q_learning_demo", maze, seed, optimal_path_length=opt, hyperparams={
        "alpha": alpha, "gamma": GAMMA, "epsilon_start": eps_start, "epsilon_end": eps_end,
        "epsilon_decay": eps_decay, "max_episodes": n_episodes})
    eps, updates = eps_start, 0
    for ep in range(n_episodes):
        s, done, G, t, hits = env.reset(), False, 0.0, 0, 0
        while not done:
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(Q[s].argmax())
            s2, r, term, trunc, info = env.step(a)
            Q[s, a] += alpha * (r + GAMMA * (0 if term else Q[s2].max()) - Q[s, a])
            updates += 1; G += r; t += 1; hits += info["hit_hazard"]
            s, done = s2, term or trunc
        eps = max(eps_end, eps * eps_decay)
        log.log_episode(ep, env.total_steps, updates, G, t, term, hits,
                        greedy=evaluate_greedy(env, Q), epsilon=eps)
    return log.finish(q_table_entries=Q.size,
                      optimal_path_length_after_change=after,
                      change_at_step=getattr(env, "change_at_step", None))

if __name__ == "__main__":
    for maze in ["small", "cliff"]:
        for seed in SEEDS[:3]:
            s = run_q_learning(maze, seed)
            print(maze, seed, "gap:", s["path_gap"], "stable at steps:", s["env_steps_to_stable_optimal"])
