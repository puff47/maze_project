"""
results_logger.py - everyone logs through this so the CSVs have identical columns.

Usage (one RunLogger per method x maze x seed):

    log = RunLogger("q_learning", "small", seed, hyperparams={"alpha": 0.1, "epsilon_start": 1.0},
                    optimal_path_length=optimal_path_length(env))
    for episode in range(n_episodes):
        ...train one episode...
        log.log_episode(episode, env_steps=env.total_steps, updates=n_updates,
                        episode_return=G, episode_steps=t, reached_goal=reached, hazard_hits=h,
                        greedy=evaluate_greedy(env, Q), epsilon=eps)
    log.finish(q_table_entries=Q.size)

Files written (each teammate only ever writes their own method's files, so Git never conflicts):
    results/episodes/<method>/<maze>_seed<k>.csv   one row per episode (or per sweep for Value Iteration)
    results/summary_<method>.csv                    one row per run (appended)
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

EPISODE_COLUMNS = [
    "run_id", "method", "maze", "seed",
    "episode",              # episode number (Value Iteration: sweep number)
    "env_steps",            # cumulative REAL environment steps (Value Iteration: 0)
    "updates",              # cumulative value updates / Bellman backups, incl. Dyna-Q planning
    "episode_return", "episode_steps", "reached_goal", "hazard_hits",
    "greedy_reached_goal", "greedy_path_length", "greedy_return",   # from evaluate_greedy()
    "epsilon", "wall_clock_sec",
]

SUMMARY_COLUMNS = [
    "run_id", "method", "maze", "seed",
    # hyperparameters
    "alpha", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay", "planning_steps_n", "theta",
    "max_episodes", "extra_params",
    # quality of the final policy
    "optimal_path_length", "final_path_length", "path_gap", "final_reached_goal", "final_return",
    # sample efficiency
    "episodes_to_first_optimal", "env_steps_to_first_optimal",
    "episodes_to_stable_optimal", "env_steps_to_stable_optimal",
    # cost
    "total_env_steps", "total_updates", "wall_clock_sec", "q_table_entries", "model_entries",
    # Value Iteration only
    "sweeps_to_converge", "final_delta",
    # changing-maze test
    "optimal_path_length_after_change", "env_steps_to_recover_after_change",
    # misc
    "hazard_hits_total", "code_version", "notes",
]

_HYPERPARAM_COLUMNS = {"alpha", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay",
                       "planning_steps_n", "theta", "max_episodes"}


def _git_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return ""


class RunLogger:
    def __init__(self, method: str, maze: str, seed: int, hyperparams: dict | None = None,
                 optimal_path_length: int | None = None, results_dir: Path = RESULTS_DIR,
                 variant: str | None = None):
        self.method, self.maze, self.seed = method, maze, seed
        self.hp = dict(hyperparams or {})
        # `variant` keeps runs of the same method/maze/seed apart (e.g. Dyna-Q planning steps).
        # If not given, it is derived from planning_steps_n, so n=0, 5, 20, 50 never overwrite each other.
        if variant is None and "planning_steps_n" in self.hp:
            variant = f"n{self.hp['planning_steps_n']}"
        self.variant = variant
        self.opt = optimal_path_length
        self.results_dir = Path(results_dir)
        tag = f"{variant}_" if variant else ""
        self.run_id = f"{method}_{maze}_{tag}seed{seed}"
        self.rows: list[dict] = []
        self._t0 = time.perf_counter()

    def log_episode(self, episode: int, env_steps: int, updates: int, episode_return=None,
                    episode_steps=None, reached_goal=None, hazard_hits=None, greedy: dict | None = None,
                    epsilon=None) -> None:
        g = greedy or {}
        self.rows.append({
            "run_id": self.run_id, "method": self.method, "maze": self.maze, "seed": self.seed,
            "episode": episode, "env_steps": env_steps, "updates": updates,
            "episode_return": episode_return, "episode_steps": episode_steps,
            "reached_goal": reached_goal, "hazard_hits": hazard_hits,
            "greedy_reached_goal": g.get("reached_goal"), "greedy_path_length": g.get("path_length"),
            "greedy_return": g.get("total_return"), "epsilon": epsilon,
            "wall_clock_sec": round(time.perf_counter() - self._t0, 4),
        })

    # ---------- derived metrics ----------
    def _is_opt(self, row, opt) -> bool:
        return opt is not None and row["greedy_path_length"] is not None and row["greedy_path_length"] == opt

    def _first_and_stable(self, rows, opt):
        flags = [self._is_opt(r, opt) for r in rows]
        first = next((i for i, f in enumerate(flags) if f), None)
        stable = None
        if flags and flags[-1]:
            stable = len(flags) - 1
            while stable > 0 and flags[stable - 1]:
                stable -= 1
        return first, stable

    def finish(self, optimal_path_length_after_change: int | None = None,
               change_at_step: int | None = None, **extra) -> dict:
        """Write the episode CSV, append the summary row, return the summary dict.

        extra can include: q_table_entries, model_entries, sweeps_to_converge, final_delta,
        total_updates, total_env_steps, notes.
        """
        rows = self.rows
        last = rows[-1] if rows else {}
        # With a changing maze, "first optimal" is measured against the layout BEFORE the change.
        pre = [r for r in rows if change_at_step is None or r["env_steps"] < change_at_step] or rows
        first, stable = self._first_and_stable(pre, self.opt)
        steps_of = lambda idx: pre[idx]["env_steps"] if idx is not None else None
        eps_of = lambda idx: pre[idx]["episode"] if idx is not None else None

        recover = None
        if change_at_step is not None and optimal_path_length_after_change is not None:
            post = [r for r in rows if r["env_steps"] >= change_at_step]
            f_post, _ = self._first_and_stable(post, optimal_path_length_after_change)
            if f_post is not None:
                recover = post[f_post]["env_steps"] - change_at_step

        final_len = last.get("greedy_path_length")
        # The final policy is evaluated on the FINAL layout, so compare with that layout's optimum.
        changed_by_end = (change_at_step is not None and optimal_path_length_after_change is not None
                          and last.get("env_steps", 0) >= change_at_step)
        final_opt = optimal_path_length_after_change if changed_by_end else self.opt
        known = {k: self.hp.pop(k) for k in list(self.hp) if k in _HYPERPARAM_COLUMNS}
        summary = {c: "" for c in SUMMARY_COLUMNS}
        summary.update(known)
        summary.update({
            "run_id": self.run_id, "method": self.method, "maze": self.maze, "seed": self.seed,
            "extra_params": json.dumps(self.hp) if self.hp else "",
            "optimal_path_length": self.opt,
            "final_path_length": final_len,
            "path_gap": (final_len - final_opt) if (final_len is not None and final_opt is not None) else None,
            "final_reached_goal": last.get("greedy_reached_goal"),
            "final_return": last.get("greedy_return"),
            "episodes_to_first_optimal": eps_of(first), "env_steps_to_first_optimal": steps_of(first),
            "episodes_to_stable_optimal": eps_of(stable), "env_steps_to_stable_optimal": steps_of(stable),
            "total_env_steps": last.get("env_steps"), "total_updates": last.get("updates"),
            "wall_clock_sec": round(time.perf_counter() - self._t0, 4),
            "optimal_path_length_after_change": optimal_path_length_after_change,
            "env_steps_to_recover_after_change": recover,
            "hazard_hits_total": sum(r["hazard_hits"] or 0 for r in rows),
            "code_version": _git_version(),
        })
        summary.update(extra)

        ep_dir = self.results_dir / "episodes" / self.method
        ep_dir.mkdir(parents=True, exist_ok=True)
        with open(ep_dir / f"{self.maze}_{self.variant + '_' if self.variant else ''}seed{self.seed}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
            w.writeheader()
            w.writerows(rows)

        path = self.results_dir / f"summary_{self.method}.csv"
        new_file = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            if new_file:
                w.writeheader()
            w.writerow(summary)
        return summary


def load_all_summaries(results_dir: Path = RESULTS_DIR):
    """Merge every teammate's summary_<method>.csv into one DataFrame (for plots and tables)."""
    import pandas as pd
    files = sorted(Path(results_dir).glob("summary_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=SUMMARY_COLUMNS)


def load_all_episodes(results_dir: Path = RESULTS_DIR):
    import pandas as pd
    files = sorted(Path(results_dir).glob("episodes/*/*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=EPISODE_COLUMNS)


def write_templates(folder: Path = Path(__file__).parent) -> None:
    """Regenerate the header-only template CSVs so they always match the column lists above."""
    for name, cols in [("results_summary_template.csv", SUMMARY_COLUMNS),
                       ("episode_log_template.csv", EPISODE_COLUMNS)]:
        with open(Path(folder) / name, "w", newline="") as f:
            csv.writer(f).writerow(cols)


if __name__ == "__main__":
    write_templates()
    print("Templates written.")
