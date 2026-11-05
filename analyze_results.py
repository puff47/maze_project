"""
analyze_results.py - merges everyone's results and produces the figures and tables for the report/slides.

Put all three methods' files in results/ first:
    results/summary_value_iteration.csv, summary_q_learning.csv, summary_dyna_q.csv
    results/episodes/<method>/*.csv
Then run:   python analyze_results.py
Output:     analysis/figures/*.png   and   analysis/tables/*.csv + results_tables.md

Needs: numpy, pandas, matplotlib, scipy.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from maze_env import make_env, load_layout, optimal_path_length, CHANGE_AT_STEP, ACTION_ARROWS
from results_logger import RESULTS_DIR, load_all_summaries
from value_iteration import value_iteration

OUT = Path(__file__).parent / "analysis"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

MAZES = ["small", "large", "cliff", "change"]
MAZE_TITLES = {"small": "Small maze (9x10)", "large": "Large maze (21x21)",
               "cliff": "Cliff maze (6x14)", "change": "Changing maze (7x12)"}
VI, QL = "Value Iteration", "Q-learning"
DYNA = ["Dyna-Q (n=5)", "Dyna-Q (n=20)", "Dyna-Q (n=50)"]
LEARNERS = [QL] + DYNA
COLORS = {VI: "#555555", QL: "#1f77b4", "Dyna-Q (n=0)": "#bbbbbb",
          "Dyna-Q (n=5)": "#fdae6b", "Dyna-Q (n=20)": "#f16913", "Dyna-Q (n=50)": "#a63603"}
OPT = {m: optimal_path_length(make_env(m)) for m in ["small", "large", "cliff"]}
OPT_CHANGE_BEFORE = optimal_path_length(load_layout("change_before"))
OPT_CHANGE_AFTER = optimal_path_length(load_layout("change_after"))

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})


#  loading data
def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["method"].isin(["value_iteration", "q_learning", "dyna_q"])].copy()
    df["n"] = pd.to_numeric(df["run_id"].str.extract(r"_n(\d+)_seed")[0])
    df["label"] = np.select(
        [df.method == "value_iteration", df.method == "q_learning"],
        [VI, QL], default="Dyna-Q (n=" + df["n"].fillna(-1).astype(int).astype(str) + ")")
    return df


def load_episodes(results_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(results_dir / "episodes" / "*" / "*.csv")))
    cols = ["run_id", "method", "maze", "seed", "env_steps", "greedy_path_length"]
    frames = [pd.read_csv(f, usecols=cols) for f in files]
    ep = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return add_label(ep)


# stats helpers
def ci95(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, max(m - h, 0.0), m + h


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def fmt_num(x, digits=0):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{digits}f}"


def to_md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.to_numpy()]
    return "\n".join(lines)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# Creating learning curves
def optimal_flags(g: pd.DataFrame, grid: np.ndarray, maze: str) -> np.ndarray:
    """For each x in grid: is the greedy policy (as last evaluated at or before x) optimal for the CURRENT layout?"""
    steps = g["env_steps"].to_numpy()
    plen = g["greedy_path_length"].to_numpy(float)
    idx = np.searchsorted(steps, grid, side="right") - 1
    ok = idx >= 0
    out = np.zeros(len(grid), bool)
    if maze == "change":
        target = np.where(grid[ok] < CHANGE_AT_STEP, OPT_CHANGE_BEFORE, OPT_CHANGE_AFTER)
    else:
        target = OPT[maze]
    out[ok] = plen[idx[ok]] == target      # NaN (goal not reached) compares False
    return out


def proportion_curve(ep: pd.DataFrame, maze: str, label: str, grid: np.ndarray):
    runs = [optimal_flags(g, grid, maze) for _, g in ep[(ep.maze == maze) & (ep.label == label)].groupby("run_id")]
    if not runs:
        return None
    M = np.vstack(runs)
    k = M.sum(axis=0)
    lo, hi = wilson(k, M.shape[0])
    return M.mean(axis=0), lo, hi, M.shape[0]


def plot_curves(ax, ep, maze, xmax, labels):
    grid = np.unique(np.geomspace(100, xmax, 300))
    for lab in labels:
        res = proportion_curve(ep, maze, lab, grid)
        if res is None:
            continue
        p, lo, hi, n = res
        ax.plot(grid, p, color=COLORS[lab], lw=2.2, label=f"{lab}")
        ax.fill_between(grid, lo, hi, color=COLORS[lab], alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("real environment steps (log scale)")
    ax.set_title(MAZE_TITLES[maze])


def fig_learning_curves(ep, summ):
    avail = [l for l in LEARNERS if (ep.label == l).any()]
    if not avail:
        return
    mazes = ["small", "large", "cliff"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, maze in zip(axes, mazes):
        xmax = ep[(ep.maze == maze) & (ep.label.isin(avail))].groupby("run_id")["env_steps"].max().median()
        plot_curves(ax, ep, maze, xmax, avail)
    axes[0].set_ylabel("share of runs whose greedy path is optimal")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Learning speed: how fast does each method find the optimal path? (shaded = 95% CI over seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_change(ep, summ):
    avail = [l for l in LEARNERS if (ep.label == l).any()]
    if not avail:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    xmax = ep[(ep.maze == "change") & (ep.label.isin(avail))].groupby("run_id")["env_steps"].max().median()
    plot_curves(a1, ep, "change", max(xmax, CHANGE_AT_STEP * 1.5), avail)
    a1.axvline(CHANGE_AT_STEP, color="k", ls="--", lw=1.2)
    a1.text(CHANGE_AT_STEP * 1.03, 0.5, "shortcut opens\n(path 22 -> 4)", fontsize=10, va="center")
    a1.set_ylabel("share of runs optimal for the current maze")
    a1.legend(loc="lower left", frameon=False)
    a1.set_title("Changing maze: before and after the shortcut opens")

    s = summ[summ.maze == "change"]
    vals = []
    for lab in avail:
        d = s[s.label == lab]
        if len(d):
            vals.append((lab, (d.path_gap == 0).sum(), len(d)))
    for i, (lab, k, n) in enumerate(vals):
        a2.bar(i, k / n, color=COLORS[lab])
        a2.text(i, k / n + 0.02, f"{k}/{n}", ha="center")
    a2.set_xticks(range(len(vals)))
    a2.set_xticklabels([v[0].replace(" (", "\n(") for v in vals], fontsize=10)
    a2.set_ylim(0, 1.1)
    a2.set_ylabel("share of runs using the shortcut at the end")
    a2.set_title("Did the agent find the new shortcut?")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_changing_maze.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# bars and ablation
def fig_steps_bars(summ):
    avail = [l for l in LEARNERS if (summ.label == l).any()]
    if not avail:
        return
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    for ax, maze in zip(axes, MAZES):
        for i, lab in enumerate(avail):
            d = summ[(summ.maze == maze) & (summ.label == lab)]
            if d.empty:
                continue
            reached = d["env_steps_to_stable_optimal"].dropna()
            m, lo, hi = ci95(reached)
            if not np.isnan(m):
                ax.bar(i, m, color=COLORS[lab], yerr=None if np.isnan(lo) else [[m - lo], [hi - m]], capsize=4)
            top = hi if not np.isnan(hi) else m
            ax.text(i, (top if not np.isnan(top) else 1) * 1.12, f"{len(reached)}/{len(d)}", ha="center", fontsize=10)
        ax.set_yscale("log")
        ax.set_xticks(range(len(avail)))
        ax.set_xticklabels([a.replace(" (", "\n(") for a in avail], fontsize=9)
        ax.set_title(MAZE_TITLES[maze] + ("\n(before the change)" if maze == "change" else ""), fontsize=11)
    axes[0].set_ylabel("real steps until the optimal path is kept")
    fig.suptitle("Sample efficiency (mean over runs that reached it, 95% CI; label = runs that reached it)", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_steps_to_optimal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(summ):
    d = summ[summ.method == "dyna_q"]
    if d.empty:
        return
    ns = sorted(d["n"].dropna().astype(int).unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    palette = {"small": "#1f77b4", "large": "#2ca02c", "cliff": "#d62728", "change": "#9467bd"}
    for maze in MAZES:
        xs, ys, es, us = [], [], [], []
        for n in ns:
            x = d[(d.maze == maze) & (d.n == n)]
            if x.empty:
                continue
            m, lo, hi = ci95(x["env_steps_to_stable_optimal"])
            xs.append(n); ys.append(m); es.append(0 if np.isnan(lo) else (hi - lo) / 2)
            us.append(x["total_updates"].mean())
        a1.errorbar(range(len(xs)), ys, yerr=es, marker="o", lw=2, capsize=4, color=palette[maze], label=maze)
        a2.plot(range(len(xs)), us, marker="o", lw=2, color=palette[maze], label=maze)
    for ax in (a1, a2):
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(ns)
        ax.set_xlabel("planning steps per real step (n)")
        ax.set_yscale("log")
    a1.set_ylabel("real steps until the optimal path is kept")
    a1.set_title("More planning -> fewer real steps")
    a2.set_ylabel("total value updates")
    a2.set_title("...but many more updates (computation)")
    a1.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_planning_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Value Iteration visuals
def draw_values(ax, env, V, Q=None, title=""):
    grid = V.reshape(env.n_rows, env.n_cols).astype(float).copy()
    for s in env.walls:
        grid[divmod(s, env.n_cols)] = np.nan
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#222222")
    ax.imshow(grid, cmap=cmap, vmin=-100, vmax=100)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in range(env.n_states):
        r, c = divmod(s, env.n_cols)
        if s == env.goal:
            ax.text(c, r, "G", ha="center", va="center", color="white", fontweight="bold")
        elif s in env.hazards:
            ax.text(c, r, "H", ha="center", va="center", color="red", fontweight="bold")
        elif s == env.start:
            tag = "S" + (ACTION_ARROWS_UNI[int(Q[s].argmax())] if Q is not None else "")
            ax.text(c, r, tag, ha="center", va="center", color="white", fontweight="bold")
        elif Q is not None and s not in env.walls:
            ax.text(c, r, ACTION_ARROWS_UNI[int(Q[s].argmax())], ha="center", va="center", color="white", fontsize=11)
    ax.set_title(title, fontsize=11)


ACTION_ARROWS_UNI = ["\u2191", "\u2193", "\u2190", "\u2192"]


def fig_vi_visuals():
    env = make_env("small")
    snaps, wanted = {}, {1, 5, 10}

    def cb(sweep, V, Q, updates):
        if sweep in wanted:
            snaps[sweep] = V.copy()

    V, Q, sweeps, _ = value_iteration(env, on_sweep=cb)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, sw in zip(axes, [1, 5, 10]):
        draw_values(ax, env, snaps[sw], title=f"after sweep {sw}")
    draw_values(axes[3], env, V, Q, title=f"converged (sweep {sweeps}) + policy")
    fig.suptitle("Value Iteration on the small maze: value spreads backwards from the goal, one step per sweep", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_vi_value_spread.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
    for ax, maze in zip(axes, ["small", "cliff"]):
        e = make_env(maze)
        V, Q, sw, _ = value_iteration(e)
        draw_values(ax, e, V, Q, title=f"{MAZE_TITLES[maze]}: optimal values and policy ({sw} sweeps)")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_vi_policies.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# tables
def table_main(summ):
    rows = []
    for maze in MAZES:
        for lab in [VI] + LEARNERS:
            d = summ[(summ.maze == maze) & (summ.label == lab)]
            if d.empty:
                continue
            N = len(d)
            reached = d["env_steps_to_stable_optimal"].dropna()
            if lab == VI:
                steps = "0 (model given)"
            else:
                m, lo, hi = ci95(reached)
                steps = "never" if np.isnan(m) else f"{m:,.0f}" + ("" if np.isnan(lo) else f" [{lo:,.0f}, {hi:,.0f}]")
            rows.append({
                "Maze": maze, "Method": lab,
                "Final path optimal": f"{(d.path_gap == 0).sum()}/{N}",
                "Mean path gap": fmt_num(d.path_gap.mean(), 2),
                "Runs reaching & keeping optimum": "-" if lab == VI else f"{len(reached)}/{N}",
                "Real steps to keep optimum (mean [95% CI])": steps,
                "Total updates (mean)": fmt_num(d.total_updates.mean()),
                "Time, s (mean)": fmt_num(d.wall_clock_sec.mean(), 2),
                "Sweeps (VI)": fmt_num(d.sweeps_to_converge.mean(), 0) if lab == VI else "-",
                "Hazard hits per run": fmt_num(d.hazard_hits_total.mean(), 0) if maze == "cliff" and lab != VI else "-",
            })
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table1_main_results.csv", index=False)
    return df


def table_significance(summ):
    q = summ[summ.label == QL]
    rows = []
    for maze in MAZES:
        a = q[q.maze == maze]
        if a.empty:
            continue
        xa = a["env_steps_to_stable_optimal"].fillna(a["total_env_steps"]).to_numpy(float)   # censored at run length
        for lab in DYNA:
            b = summ[(summ.maze == maze) & (summ.label == lab)]
            if b.empty:
                continue
            xb = b["env_steps_to_stable_optimal"].fillna(b["total_env_steps"]).to_numpy(float)
            u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            rows.append({"Maze": maze, "Comparison": f"{QL} vs {lab}",
                         "Median steps, Q-learning": fmt_num(np.median(xa)),
                         "Median steps, Dyna-Q": fmt_num(np.median(xb)),
                         "Effect size (negative = Dyna-Q needs fewer steps)": round(1 - 2 * u / (len(xa) * len(xb)), 2),
                         "p (Mann-Whitney U)": p})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["p (Holm-adjusted)"] = holm(df["p (Mann-Whitney U)"])
    df["Significant (adj. p < 0.05)"] = np.where(df["p (Holm-adjusted)"] < 0.05, "yes", "no")
    for c in ["p (Mann-Whitney U)", "p (Holm-adjusted)"]:
        df[c] = df[c].map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}")
    df.to_csv(TAB / "table2_significance.csv", index=False)
    return df


def table_change(summ):
    rows = []
    for lab in LEARNERS:
        d = summ[(summ.maze == "change") & (summ.label == lab)]
        if d.empty:
            continue
        N = len(d)
        pre = d["env_steps_to_stable_optimal"].dropna()
        rec = d["env_steps_to_recover_after_change"].dropna()
        rows.append({"Method": lab,
                     "Optimal path (22) found before change": f"{len(pre)}/{N}",
                     "Ever found the shortcut (4)": f"{len(rec)}/{N}",
                     "Using shortcut at the end": f"{(d.path_gap == 0).sum()}/{N}",
                     "Median real steps to find it (after change)": fmt_num(rec.median()) if len(rec) else "-"})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table3_changing_maze.csv", index=False)
    return df


def table_ablation(summ):
    d = summ[summ.method == "dyna_q"]
    rows = []
    for maze in MAZES:
        for n in sorted(d["n"].dropna().astype(int).unique()):
            x = d[(d.maze == maze) & (d.n == n)]
            if x.empty:
                continue
            reached = x["env_steps_to_stable_optimal"].dropna()
            rows.append({"Maze": maze, "n": n, "Reached & kept optimum": f"{len(reached)}/{len(x)}",
                         "Real steps (mean)": fmt_num(reached.mean()), "Total updates (mean)": fmt_num(x.total_updates.mean()),
                         "Time, s (mean)": fmt_num(x.wall_clock_sec.mean(), 2),
                         "Model entries (mean)": fmt_num(x.model_entries.mean(), 0)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table4_planning_ablation.csv", index=False)
    return df


# Main Method
def main(results_dir: Path = RESULTS_DIR):
    summ = add_label(load_all_summaries(results_dir))
    ep = load_episodes(results_dir)
    print("Runs per method/maze:\n", summ.groupby(["label", "maze"]).size().unstack(fill_value=0), "\n")
    if summ.empty:
        sys.exit("No results found in " + str(results_dir))

    fig_learning_curves(ep, summ)
    fig_change(ep, summ)
    fig_steps_bars(summ)
    fig_ablation(summ)
    fig_vi_visuals()

    sections = [("Table 1. Main results (all mazes, 30 seeds)", table_main(summ)),
                ("Table 2. Q-learning vs Dyna-Q: steps to keep the optimal path (runs that never did are counted at their run length)", table_significance(summ)),
                ("Table 3. Changing maze", table_change(summ)),
                ("Table 4. Planning-steps ablation (Dyna-Q)", table_ablation(summ))]
    md = []
    for title, df in sections:
        if df is not None and len(df):
            md.append(f"### {title}\n\n{to_md(df)}\n")
            print(title); print(df.to_string(index=False)); print()
    (TAB / "results_tables.md").write_text("\n".join(md))
    print("Figures ->", FIG, "\nTables  ->", TAB)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR)
