"""Cross-agent statistical comparison, paired by seed.

Because every agent faces the identical return path for seed *k*, market
luck differences out: the honest comparison is the distribution of
*per-seed paired differences*. We report the mean paired difference with a
bootstrap 95% CI (resampling seeds), the per-seed win rate, and flag
significance only when the CI excludes zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 12345  # fixed: comparisons are reproducible


@dataclass
class RunData:
    """One agent's per-seed metrics, as loaded from a run directory."""

    name: str
    scenario: str
    metrics_by_seed: dict[int, dict[str, Any]]

    @property
    def seeds(self) -> set[int]:
        return set(self.metrics_by_seed)

    def values(self, key: str, seeds: Optional[list[int]] = None) -> np.ndarray:
        seeds = sorted(self.seeds) if seeds is None else seeds
        return np.array([float(self.metrics_by_seed[s][key]) for s in seeds])

    def fraction(self, key: str) -> float:
        vals = [bool(self.metrics_by_seed[s][key]) for s in sorted(self.seeds)]
        return float(np.mean(vals)) if vals else float("nan")


@dataclass
class PairedResult:
    agent_a: str
    agent_b: str
    metric: str
    n_seeds: int
    mean_diff: float           # a − b
    ci_lo: float
    ci_hi: float
    win_rate_a: float          # P(a > b) across seeds
    significant: bool

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def paired_comparison(a: RunData, b: RunData, metric: str = "ce_wealth",
                      n_boot: int = BOOTSTRAP_ITERS) -> PairedResult:
    """Paired (by shared seed) comparison of one metric between two runs."""
    common = sorted(a.seeds & b.seeds)
    if not common:
        raise ValueError(f"runs {a.name} and {b.name} share no seeds")
    diffs = a.values(metric, common) - b.values(metric, common)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return PairedResult(
        agent_a=a.name, agent_b=b.name, metric=metric, n_seeds=len(common),
        mean_diff=float(diffs.mean()), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
        win_rate_a=float(np.mean(diffs > 0)),
        significant=bool(ci_lo > 0 or ci_hi < 0),
    )


# ---------------------------------------------------------------------------
# Aggregate summary rows
# ---------------------------------------------------------------------------

SUMMARY_SPEC = [
    ("median_ce_wealth", "ce_wealth", "median"),
    ("p10_ce_wealth", "ce_wealth", "p10"),
    ("median_ce_annual", "ce_annual", "median"),
    ("median_terminal_wealth_real", "terminal_wealth_real", "median"),
    ("mean_lifetime_taxes_real", "lifetime_taxes_real", "mean"),
    ("mean_lifetime_penalties_real", "lifetime_penalties_real", "mean"),
    ("mean_violations", "violations", "mean"),
    ("mean_decision_churn", "decision_churn", "mean"),
]

FRACTION_SPEC = [
    ("p_ruin", "ruin"),
    ("p_all_goals_met", "all_goals_met"),
    ("p_college_goal_met", "college_goal_met"),
    ("p_retirement_goal_met", "retirement_goal_met"),
]


def summarize_run(run: RunData) -> dict[str, float]:
    out: dict[str, float] = {"n_seeds": len(run.seeds)}
    for label, key, how in SUMMARY_SPEC:
        vals = run.values(key)
        if how == "median":
            out[label] = float(np.median(vals))
        elif how == "p10":
            out[label] = float(np.percentile(vals, 10))
        else:
            out[label] = float(np.mean(vals))
    for label, key in FRACTION_SPEC:
        out[label] = run.fraction(key)
    return out


def compare_runs(runs: list[RunData], metric: str = "ce_wealth",
                 reference: Optional[str] = None,
                 ) -> dict[str, Any]:
    """Full comparison payload: per-run summaries + pairwise stats against a
    reference run (the rule-based ``expert`` if present, else the first)."""
    if reference is None:
        names = [r.name for r in runs]
        reference = "expert" if "expert" in names else names[0]
    ref = next(r for r in runs if r.name == reference)
    summaries = {r.name: summarize_run(r) for r in runs}
    paired = {}
    for r in runs:
        if r.name == ref.name:
            continue
        paired[r.name] = paired_comparison(r, ref, metric).to_dict()
    return {"reference": ref.name, "metric": metric,
            "summaries": summaries, "paired_vs_reference": paired}


def format_comparison_text(payload: dict[str, Any]) -> str:
    """Plain-text table for the CLI."""
    summaries: dict[str, dict] = payload["summaries"]
    names = list(summaries)
    lines = []
    hdr = f"{'metric':<34}" + "".join(f"{n:>18}" for n in names)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    rows = [
        ("seeds", "n_seeds", "{:.0f}"),
        ("median CE wealth (real $)", "median_ce_wealth", "{:,.0f}"),
        ("p10 CE wealth (real $)", "p10_ce_wealth", "{:,.0f}"),
        ("median CE annual (real $/yr)", "median_ce_annual", "{:,.0f}"),
        ("median terminal wealth (real $)", "median_terminal_wealth_real", "{:,.0f}"),
        ("P(ruin before 95)", "p_ruin", "{:.1%}"),
        ("P(all goals met)", "p_all_goals_met", "{:.1%}"),
        ("P(college goal met)", "p_college_goal_met", "{:.1%}"),
        ("P(retirement goal met)", "p_retirement_goal_met", "{:.1%}"),
        ("mean lifetime taxes (real $)", "mean_lifetime_taxes_real", "{:,.0f}"),
        ("mean lifetime penalties (real $)", "mean_lifetime_penalties_real", "{:,.0f}"),
        ("mean violations / trial", "mean_violations", "{:.2f}"),
        ("mean decision churn", "mean_decision_churn", "{:.3f}"),
    ]
    for label, key, fmt in rows:
        lines.append(f"{label:<34}"
                     + "".join(f"{fmt.format(summaries[n][key]):>18}" for n in names))
    lines.append("")
    ref = payload["reference"]
    metric = payload["metric"]
    lines.append(f"Paired differences vs '{ref}' on {metric} "
                 f"(positive = better than {ref}; bootstrap 95% CI):")
    for name, p in payload["paired_vs_reference"].items():
        sig = "SIGNIFICANT" if p["significant"] else "not significant"
        lines.append(f"  {name:<12} mean diff {p['mean_diff']:>14,.0f}   "
                     f"CI [{p['ci_lo']:,.0f}, {p['ci_hi']:,.0f}]   "
                     f"win rate {p['win_rate_a']:.1%}   ({sig}, n={p['n_seeds']})")
    return "\n".join(lines)
