"""Comparison charts (optional; requires matplotlib).

Produces a single PNG with three panels:
1. wealth fan chart (median + 10-90% band of real net worth by year, per agent)
2. CE-wealth distribution (per-seed) as box plots
3. goal-funding rates as grouped bars
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .comparison import RunData


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for charts: pip install matplotlib "
            "(or finplan-arena[charts])") from e


def comparison_charts(runs: list[RunData], out_path: str,
                      title: Optional[str] = None) -> str:
    plt = _require_matplotlib()
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(title or f"FinPlan Arena — scenario '{runs[0].scenario}'",
                 fontsize=13)

    # -- 1. wealth fan chart -------------------------------------------------
    ax = axes[0]
    for i, run in enumerate(runs):
        seeds = sorted(run.seeds)
        trajs = [run.metrics_by_seed[s]["trajectory_net_worth_real"]
                 for s in seeds]
        n = min(len(t) for t in trajs)
        m = np.array([t[:n] for t in trajs]) / 1e6
        yrs = np.arange(n)
        med = np.median(m, axis=0)
        p10, p90 = np.percentile(m, [10, 90], axis=0)
        c = colors[i % len(colors)]
        ax.plot(yrs, med, color=c, label=run.name, lw=2)
        ax.fill_between(yrs, p10, p90, color=c, alpha=0.15)
    ax.set_xlabel("sim year")
    ax.set_ylabel("real net worth ($M)")
    ax.set_title("Net worth (median, 10–90% band)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # -- 2. CE wealth distributions --------------------------------------------
    ax = axes[1]
    data = [run.values("ce_wealth") / 1e6 for run in runs]
    bp = ax.boxplot(data, tick_labels=[r.name for r in runs], showmeans=True,
                    patch_artist=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i % len(colors)])
        box.set_alpha(0.4)
    ax.set_ylabel("CE wealth ($M, real)")
    ax.set_title("Certainty-equivalent wealth by seed")
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=20)

    # -- 3. goal funding rates ----------------------------------------------------
    ax = axes[2]
    goal_keys = [("no ruin", "ruin", True),   # invert
                 ("college", "college_goal_met", False),
                 ("retirement", "retirement_goal_met", False),
                 ("all goals", "all_goals_met", False)]
    width = 0.8 / len(runs)
    x = np.arange(len(goal_keys))
    for i, run in enumerate(runs):
        vals = []
        for _, key, invert in goal_keys:
            f = run.fraction(key)
            vals.append(1.0 - f if invert else f)
        ax.bar(x + i * width, vals, width, label=run.name,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x + width * (len(runs) - 1) / 2)
    ax.set_xticklabels([g[0] for g in goal_keys])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of seeds")
    ax.set_title("Goal funding rates")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
