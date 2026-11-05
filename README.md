# Maze project: shared kit

| File | Purpose |
|---|---|
| `maze_env.py` | The shared MDP (the "contract"), 4 mazes, BFS optimal-path benchmark, `evaluate_greedy()` |
| `results_logger.py` | `RunLogger` plus loaders that merge everyone's CSVs |
| `results_summary_template.csv` | Header for the one-row-per-run file |
| `episode_log_template.csv` | Header for the one-row-per-episode file |
| `example_usage.py` | Minimal Q-learning demo showing how to plug a method in (replace it with your own) |
| `mazes/*.txt` | `small`, `large`, `cliff`, `change_before` / `change_after`. Legend: `#` wall, `.` free, `S` start, `G` goal, `H` hazard |

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
