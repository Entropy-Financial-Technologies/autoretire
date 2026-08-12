"""Per-trial metrics.

Headline metric: **certainty-equivalent (CE) wealth** under CRRA utility over
annual real consumption plus a terminal bequest.

Definitions (all in start-year real dollars):

* consumption ``c_t`` = essential paid + discretionary paid that year
  (mortgage P&I and college bills are obligations, not lifestyle
  consumption; failing them shows up in shortfall/goal metrics instead).
* utility ``u(c) = c^(1-γ)/(1-γ)`` (γ=3 default; γ=1 → log), consumption
  floored at ``consumption_floor`` for utility purposes so a ruin year is
  finitely catastrophic rather than -∞ (ruin is also scored separately).
* the terminal bequest ``W_T`` (real net worth at horizon end) enters via a
  De Nardi (2004)-style luxury-good bequest utility — bounded, zero at
  ``W_T = 0``, strictly increasing:

      v(W) = θ · [ (W + κ)^(1-γ) − κ^(1-γ) ] / (1-γ)

  θ = ``bequest_weight``, κ = ``bequest_luxury_shift``. The shift keeps the
  marginal utility of the first bequest dollar finite, so leaving nothing is
  merely a missed bonus — NOT twenty phantom starvation-years (an earlier
  annuitized formulation had exactly that artifact: a small bequest could
  dominate a lifetime of good consumption, inverting the ranking).
* ``U = Σ_t β^t u(c_t) + β^T v(W_T)``
* **ce_annual** solves ``Σ_t β^t u(ce_annual) = U``: the constant real
  consumption whose lifetime utility matches the realized one.
* **ce_wealth** = ce_annual × Σ β^t — the annuity price of that consumption;
  a wealth-denominated scalar, monotone in ce_annual (the headline number).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricConfig:
    gamma: float = 3.0                 # CRRA coefficient
    beta: float = 0.97                 # annual utility discount factor
    bequest_weight: float = 5.0        # θ in the De Nardi bequest term
    bequest_luxury_shift: float = 500_000.0  # κ: bequests are luxury goods
    consumption_floor: float = 5_000.0  # real $, utility floor (ruin guard)
    retirement_target_tolerance: float = 0.98  # goal met at ≥98% of target


def crra_utility(c: float, gamma: float) -> float:
    if gamma == 1.0:
        return float(np.log(c))
    return c ** (1.0 - gamma) / (1.0 - gamma)


def inverse_crra(u: float, gamma: float) -> float:
    if gamma == 1.0:
        return float(np.exp(u))
    return (u * (1.0 - gamma)) ** (1.0 / (1.0 - gamma))


def bequest_utility(wealth: float, mcfg: MetricConfig) -> float:
    """De Nardi-style luxury bequest utility: 0 at wealth 0, bounded above,
    strictly increasing. Never punishes — a bequest is a bonus."""
    w = max(0.0, wealth)
    g, k = mcfg.gamma, mcfg.bequest_luxury_shift
    if g == 1.0:
        return mcfg.bequest_weight * float(np.log((w + k) / k))
    return mcfg.bequest_weight * ((w + k) ** (1.0 - g) - k ** (1.0 - g)) / (1.0 - g)


def certainty_equivalent(consumption_real: list[float], terminal_wealth_real: float,
                         mcfg: MetricConfig) -> dict[str, float]:
    """CE of a realized consumption stream + bequest. See module docstring."""
    g, b = mcfg.gamma, mcfg.beta
    floor = mcfg.consumption_floor
    stream = [max(c, floor) for c in consumption_real]

    weights = np.array([b ** t for t in range(len(stream))])
    utils = np.array([crra_utility(c, g) for c in stream])
    total_u = float(weights @ utils)
    total_u += (b ** len(stream)) * bequest_utility(terminal_wealth_real, mcfg)
    ce_annual = inverse_crra(total_u / weights.sum(), g)
    ce_wealth = ce_annual * float(weights.sum())
    return {"ce_annual": ce_annual, "ce_wealth": ce_wealth}


RUIN_TIERS = ("essential", "mortgage", "tax")


def compute_trial_metrics(logs: list[dict], final_state: dict,
                          goals: dict, mcfg: MetricConfig) -> dict[str, Any]:
    """Metrics for one (agent, seed) trial from its year logs.

    ``logs`` are YearLog dicts; ``final_state`` a HouseholdState dict;
    ``goals`` the scenario GoalSet dict."""
    cons_real = [lg["consumption_real"] for lg in logs]
    idx_final = logs[-1]["inflation_index_eoy"]
    terminal_wealth_real = logs[-1]["net_worth_nominal"] / idx_final

    out: dict[str, Any] = {}
    out.update(certainty_equivalent(cons_real, terminal_wealth_real, mcfg))
    out["terminal_wealth_real"] = terminal_wealth_real

    # ---- goal 1: essential funding / ruin ---------------------------------
    shortfall_years = 0
    shortfall_total_real = 0.0
    for lg in logs:
        year_short = sum(s["amount"] for s in lg["shortfalls"]
                         if s["tier"] in RUIN_TIERS)
        if year_short > 0.5:
            shortfall_years += 1
            shortfall_total_real += year_short / lg["inflation_index_boy"]
    out["essential_shortfall_years"] = shortfall_years
    out["essential_shortfall_real"] = shortfall_total_real
    out["ruin"] = shortfall_years > 0

    # ---- goal 2: college funding -------------------------------------------
    college_due: dict[str, float] = {}
    college_paid: dict[str, float] = {}
    for lg in logs:
        for cid, rec in lg["expenses"].get("college_by_child", {}).items():
            college_due[cid] = college_due.get(cid, 0.0) + rec["due"]
            college_paid[cid] = college_paid.get(cid, 0.0) + rec["paid"]
    frac = {cid: (college_paid[cid] / college_due[cid]) if college_due[cid] > 0
            else 1.0 for cid in college_due}
    out["college_funded_fraction"] = frac
    target = goals.get("college_fund_fraction", 0.0)
    out["college_goal_met"] = all(f >= target - 1e-6 for f in frac.values()) \
        if frac else True

    # ---- goal 3: retirement lifestyle -----------------------------------------
    retirement_cons = [lg["consumption_real"] for lg in logs if lg["all_retired"]]
    target_spend = goals.get("retirement_spending_real", 0.0)
    if retirement_cons:
        mean_ret = float(np.mean(retirement_cons))
        out["retirement_spending_mean_real"] = mean_ret
        out["retirement_spending_vs_target"] = (mean_ret / target_spend
                                                if target_spend > 0 else 1.0)
        out["retirement_goal_met"] = (mean_ret >= target_spend
                                      * mcfg.retirement_target_tolerance)
    else:
        out["retirement_spending_mean_real"] = 0.0
        out["retirement_spending_vs_target"] = 0.0
        out["retirement_goal_met"] = False

    # ---- goal 4: bequest (optional) ----------------------------------------------
    beq_target = goals.get("bequest_target_real")
    out["bequest_goal_met"] = (terminal_wealth_real >= beq_target
                               if beq_target else True)

    out["all_goals_met"] = bool(not out["ruin"] and out["college_goal_met"]
                                and out["retirement_goal_met"]
                                and out["bequest_goal_met"])

    # ---- taxes, penalties, violations ----------------------------------------------
    taxes_real = sum(lg["tax"]["total_tax"] / lg["inflation_index_boy"] for lg in logs)
    pen_real = sum(lg["tax"]["penalties"] / lg["inflation_index_boy"] for lg in logs)
    out["lifetime_taxes_real"] = taxes_real
    out["lifetime_penalties_real"] = pen_real
    out["violations"] = int(sum(len(lg["violations"]) for lg in logs))
    out["forced_liquidation_years"] = int(sum(1 for lg in logs
                                              if lg["forced_liquidation"] > 0.5))

    # ---- decision churn ---------------------------------------------------------------
    churn_vals = []
    for prev, cur in zip(logs, logs[1:]):
        pa, ca = prev["allocations_target"], cur["allocations_target"]
        deltas = []
        for aid in ca:
            if aid == "cash" or aid not in pa:
                continue
            l1 = sum(abs(ca[aid][k] - pa[aid][k]) for k in ca[aid])
            deltas.append(l1 / 2.0)  # turnover convention
        if deltas:
            churn_vals.append(float(np.mean(deltas)))
    out["decision_churn"] = float(np.mean(churn_vals)) if churn_vals else 0.0

    # ---- trajectory (compact, kept even in summary logs; enables fan charts) -----
    out["trajectory_net_worth_real"] = [
        round(lg["net_worth_nominal"] / lg["inflation_index_eoy"], 2) for lg in logs]
    out["trajectory_consumption_real"] = [round(c, 2) for c in cons_real]
    return out
