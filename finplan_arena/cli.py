"""``finplan`` CLI.

    finplan run --scenario meridian --agent expert --seeds 300 --workers 8
    finplan compare runs/meridian_expert runs/meridian_naive --charts out.png
    finplan report runs/meridian_expert --out report.md
    finplan list
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .runner import (RunConfig, build_summary, load_run, load_trial,
                     make_agent_factory, run_many, AGENT_SPECS)
from .scenarios.library import SCENARIOS, get_scenario
from .scoring.comparison import compare_runs, format_comparison_text
from .scoring.metrics import MetricConfig


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = RunConfig(
        scenario=args.scenario,
        agent=args.agent,
        seeds=args.seeds,
        seed_start=args.seed_start,
        workers=args.workers,
        out_dir=args.out,
        generator=args.generator,
        block_years=args.block_years,
        log_detail=args.log_detail,
        agent_horizon=args.agent_horizon,
        llm_config_path=args.llm_config,
        metrics=MetricConfig(gamma=args.gamma),
    )
    out = run_many(cfg)
    print(f"run complete → {out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    runs = [load_run(p) for p in args.runs]
    scenarios = {r.scenario for r in runs}
    if len(scenarios) > 1:
        print(f"WARNING: comparing runs from different scenarios: {scenarios}",
              file=sys.stderr)
    payload = compare_runs(runs, metric=args.metric, reference=args.reference)
    print(format_comparison_text(payload))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json_out}")
    if args.charts:
        from .scoring.charts import comparison_charts
        path = comparison_charts(runs, args.charts)
        print(f"wrote {path}")
    return 0


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _cmd_report(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    seeds = sorted(run.seeds)
    ce = {s: run.metrics_by_seed[s]["ce_wealth"] for s in seeds}
    worst = sorted(seeds, key=lambda s: ce[s])[: args.worst]
    best = max(seeds, key=lambda s: ce[s])
    median_seed = sorted(seeds, key=lambda s: ce[s])[len(seeds) // 2]

    lines: list[str] = []
    lines.append(f"# FinPlan Arena report — agent `{run.name}`, "
                 f"scenario `{run.scenario}`")
    lines.append("")
    from .scoring.comparison import summarize_run
    s = summarize_run(run)
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| seeds | {int(s['n_seeds'])} |")
    lines.append(f"| median CE wealth (real) | {_fmt_money(s['median_ce_wealth'])} |")
    lines.append(f"| p10 CE wealth (real) | {_fmt_money(s['p10_ce_wealth'])} |")
    lines.append(f"| median terminal wealth (real) | "
                 f"{_fmt_money(s['median_terminal_wealth_real'])} |")
    lines.append(f"| P(ruin) | {s['p_ruin']:.1%} |")
    lines.append(f"| P(all goals met) | {s['p_all_goals_met']:.1%} |")
    lines.append(f"| mean lifetime taxes (real) | "
                 f"{_fmt_money(s['mean_lifetime_taxes_real'])} |")
    lines.append(f"| mean lifetime penalties (real) | "
                 f"{_fmt_money(s['mean_lifetime_penalties_real'])} |")
    lines.append(f"| mean violations / trial | {s['mean_violations']:.2f} |")
    lines.append(f"| mean decision churn | {s['mean_decision_churn']:.3f} |")
    lines.append("")

    def _trial_section(seed: int, label: str, n_years: int = 6) -> None:
        try:
            rec = load_trial(args.run, seed)
        except FileNotFoundError:
            lines.append(f"## {label} (seed {seed}) — trial file missing")
            return
        m = rec["metrics"]
        lines.append(f"## {label} — seed {seed}")
        lines.append("")
        lines.append(f"CE wealth {_fmt_money(m['ce_wealth'])} · terminal wealth "
                     f"{_fmt_money(m['terminal_wealth_real'])} · "
                     f"shortfall years {m['essential_shortfall_years']} · "
                     f"college funded "
                     f"{ {k: round(v, 2) for k, v in m['college_funded_fraction'].items()} } · "
                     f"violations {m['violations']}")
        lines.append("")
        logs = rec.get("logs", [])
        if not logs:
            return
        lines.append("| yr | age(s) | consumption (real) | net worth (real) | "
                     "tax | events |")
        lines.append("|---|---|---|---|---|---|")
        idx = sorted({0, len(logs) // 4, len(logs) // 2, 3 * len(logs) // 4,
                      len(logs) - 1} | set(
                          i for i, lg in enumerate(logs) if lg["shortfalls"]))
        for i in sorted(idx)[: n_years + 8]:
            lg = logs[i]
            ages = "/".join(str(v) for k, v in sorted(lg.get("ages", {}).items())
                            if k.startswith("a")) or "-"
            events = []
            if lg["shortfalls"]:
                events.append("SHORTFALL: " + ", ".join(
                    f"{sf['tier']} {_fmt_money(sf['amount'])}"
                    for sf in lg["shortfalls"]))
            if lg.get("rmd_forced", 0) > 0.5:
                events.append(f"forced RMD {_fmt_money(lg['rmd_forced'])}")
            if lg.get("roth_converted", 0) > 0.5:
                events.append(f"Roth conv {_fmt_money(lg['roth_converted'])}")
            if lg["violations"]:
                events.append(f"{len(lg['violations'])} violation(s)")
            nw_real = lg["net_worth_nominal"] / lg["inflation_index_eoy"]
            lines.append(
                f"| {lg['year_index']} | {ages} | "
                f"{_fmt_money(lg['consumption_real'])} | {_fmt_money(nw_real)} | "
                f"{_fmt_money(lg['tax']['total_tax'])} | "
                f"{'; '.join(events) or '—'} |")
        lines.append("")
        rat = [lg.get("rationale", "") for lg in logs[:3] if lg.get("rationale")]
        if rat:
            lines.append(f"Early rationale: “{rat[0][:240]}”")
            lines.append("")

    _trial_section(median_seed, "Median-seed trajectory")
    _trial_section(best, "Best seed")
    for w in worst:
        _trial_section(w, "Worst-seed post-mortem")

    # violation digest across all seeds
    lines.append("## Violation digest")
    lines.append("")
    counts: dict[str, int] = {}
    for s_ in seeds:
        try:
            rec = load_trial(args.run, s_)
        except FileNotFoundError:
            continue
        for lg in rec.get("logs", []):
            for v in lg["violations"]:
                counts[v["code"]] = counts.get(v["code"], 0) + 1
    if counts:
        lines.append("| violation code | occurrences (all seeds/years) |")
        lines.append("|---|---|")
        for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {code} | {n} |")
    else:
        lines.append("No violations recorded. Clean hands.")
    lines.append("")

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print("Scenarios:")
    for name, sc in SCENARIOS.items():
        print(f"  {name:<10} {sc.description[:96]}")
    print("\nAgent specs:")
    for a in AGENT_SPECS:
        print(f"  {a}")
    return 0


def _cmd_rebuild_summary(args: argparse.Namespace) -> int:
    s = build_summary(args.run)
    print(f"rebuilt summary for {args.run}: {s['n_trials']} trials")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="finplan",
        description="FinPlan Arena — evaluate agents on multi-decade "
                    "household financial planning")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run scenario × agent × seeds")
    r.add_argument("--scenario", default="meridian", choices=sorted(SCENARIOS))
    r.add_argument("--agent", default="expert",
                   help="expert|tdf|naive|mock-llm|llm:<config.json>|<model>")
    r.add_argument("--seeds", type=int, default=300)
    r.add_argument("--seed-start", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out", default=None, help="run directory (default runs/<auto>)")
    r.add_argument("--generator", default="block_bootstrap",
                   choices=["block_bootstrap", "iid_lognormal"])
    r.add_argument("--block-years", type=int, default=5)
    r.add_argument("--log-detail", default="full", choices=["full", "summary"])
    r.add_argument("--agent-horizon", default="active_then_drawdown",
                   choices=["active_then_drawdown", "full"],
                   help="'full' keeps the agent in the loop to age 95")
    r.add_argument("--llm-config", default=None,
                   help="LLM config JSON (see README) for LLM agent specs")
    r.add_argument("--gamma", type=float, default=3.0, help="CRRA γ")
    r.set_defaults(fn=_cmd_run)

    c = sub.add_parser("compare", help="paired comparison of 2+ runs")
    c.add_argument("runs", nargs="+", help="run directories")
    c.add_argument("--metric", default="ce_wealth")
    c.add_argument("--reference", default=None,
                   help="reference agent name (default: expert if present)")
    c.add_argument("--charts", default=None, help="write PNG charts here")
    c.add_argument("--json-out", default=None, help="write comparison JSON here")
    c.set_defaults(fn=_cmd_compare)

    d = sub.add_parser("report", help="single-run deep dive (markdown)")
    d.add_argument("run", help="run directory")
    d.add_argument("--worst", type=int, default=2,
                   help="number of worst-seed post-mortems")
    d.add_argument("--out", default=None, help="write markdown here (else stdout)")
    d.set_defaults(fn=_cmd_report)

    ls = sub.add_parser("list", help="list scenarios and agent specs")
    ls.set_defaults(fn=_cmd_list)

    rs = sub.add_parser("rebuild-summary",
                        help="rebuild summary.json from trial files")
    rs.add_argument("run")
    rs.set_defaults(fn=_cmd_rebuild_summary)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
