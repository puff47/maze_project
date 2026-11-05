# Maze project: shared kit

| File | Purpose |
|---|---|
| `maze_env.py` | The shared MDP (the "contract"), 4 mazes, BFS optimal-path benchmark, `evaluate_greedy()` |
| `results_logger.py` | `RunLogger` plus loaders that merge everyone's CSVs |
| `results_summary_template.csv` | Header for the one-row-per-run file |
| `episode_log_template.csv` | Header for the one-row-per-episode file |
| `example_usage.py` | Minimal Q-learning demo showing how to plug a method in (replace it with your own) |
| `mazes/*.txt` | `small`, `large`, `cliff`, `change_before` / `change_after`. Legend: `#` wall, `.` free, `S` start, `G` goal, `H` hazard |

## The contract (edit only with the whole group's agreement)
gamma = 0.99 | step reward -1 (walls bump = stay, still -1) | goal +100 (terminal) | hazard -100 and sent back to start |
max 500 steps per episode | seeds 0-29 | deterministic moves | maze changes after 20,000 real steps (`change` maze).

Optimal path lengths (BFS): small 23, large 40, cliff 13, change 22 (before) then 4 (after).

## Rules for everyone
1. Always import the env from `maze_env.py`; never copy it.
2. Run every maze for all 30 seeds in `SEEDS`.
3. Log through `RunLogger` so column names match. Use `method` names: `value_iteration`, `q_learning`, `dyna_q`
   (Dyna-Q ablation: `dyna_q` with `planning_steps_n` set to 0, 5, 20, 50, keeping the same method name).
4. Count `env_steps` as REAL environment steps only; count `updates` as every Q/V update (Dyna-Q planning included).
5. Value Iteration: log one row per sweep (`episode` = sweep number, `env_steps` = 0, `updates` = cumulative backups,
   `greedy=evaluate_greedy(env, pi)` where `pi` is the greedy action array), and pass `sweeps_to_converge` and `final_delta` to `finish()`.
6. Dyna-Q: pass `model_entries` (number of stored (s, a) pairs) to `finish()`.
7. Changing maze: call `finish(optimal_path_length_after_change=4, change_at_step=env.change_at_step)`.
8. Each person only writes their own `results/summary_<method>.csv`, so Git never conflicts. Commit the `results/` folder.

## Key columns
- `path_gap`: final greedy path length minus the BFS optimum (0 = optimal).
- `env_steps_to_first_optimal`: first time the greedy policy had an optimal path.
- `env_steps_to_stable_optimal`: first time from which it stayed optimal until the end (more reliable, use this in the report).
- `env_steps_to_recover_after_change`: real steps after the maze change until the greedy policy found the new optimal path.
- `wall_clock_sec`: includes the per-episode greedy evaluation, so only compare it between methods that log the same way.

## Merging and plotting
```python
from results_logger import load_all_summaries, load_all_episodes
df = load_all_summaries()
df.groupby(["maze", "method"])[["env_steps_to_stable_optimal", "path_gap", "total_updates"]].agg(["mean", "std"])
```
