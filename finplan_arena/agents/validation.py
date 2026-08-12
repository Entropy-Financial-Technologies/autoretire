"""Decision validator/clipper.

Enforces *rule* feasibility (IRS limits, eligibility, phase-outs, claiming
ages) before the simulator applies a decision. Every clip is recorded as a
``Violation`` — violations are a scored signal, not just bookkeeping: an
agent that keeps proposing infeasible decisions loses on the
constraint-violations metric even when clipping rescues the plan.

Resource feasibility (spending cash the household doesn't have) is enforced
later, inside the simulator, where actual cash flow is known; those clips
produce violations too (code ``insufficient_cash``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.accounts import (HSA_MAX_AGE, SS_EARLIEST_CLAIM_AGE,
                             ContributionLimits)
from ..core.state import ASSET_CLASSES, HouseholdState
from .base import Decision


@dataclass
class Violation:
    code: str
    field: str
    message: str
    proposed: float = 0.0
    clipped_to: float = 0.0

    def to_dict(self) -> dict:
        return {"code": self.code, "field": self.field, "message": self.message,
                "proposed": self.proposed, "clipped_to": self.clipped_to}


def _clean(x: float) -> float:
    """Guard against NaN/inf sneaking in from hand-built decisions."""
    if x is None or not math.isfinite(x):
        return float("nan")
    return float(x)


def validate_decision(decision: Decision, state: HouseholdState,
                      limits: ContributionLimits, estimated_magi: float,
                      ) -> tuple[Decision, list[Violation]]:
    """Return a feasible copy of ``decision`` plus the list of clips made."""
    d = Decision.model_validate(decision.model_dump(by_alias=True))
    v: list[Violation] = []

    def clip(field_name: str, value: float, lo: float, hi: float, code: str,
             message: str) -> float:
        value = _clean(value)
        if math.isnan(value):
            v.append(Violation(code="non_finite", field=field_name,
                               message="non-finite value replaced with 0",
                               proposed=float("nan"), clipped_to=0.0))
            return 0.0
        clipped = min(max(value, lo), hi)
        if abs(clipped - value) > 0.01:
            v.append(Violation(code=code, field=field_name, message=message,
                               proposed=value, clipped_to=clipped))
        return clipped

    a1 = state.adult("a1")
    a2 = state.adult("a2")

    # ---- discretionary spending bounds -----------------------------------
    d.annual_spending_discretionary = clip(
        "annual_spending_discretionary", d.annual_spending_discretionary,
        state.discretionary_floor, state.discretionary_ceiling,
        "spending_bounds",
        f"discretionary spending must lie in [{state.discretionary_floor:,.0f}, "
        f"{state.discretionary_ceiling:,.0f}]")

    # ---- per-adult payroll contributions ----------------------------------
    c = d.contributions

    def works_this_year(pid: str) -> bool:
        """Consistent with the engine: an adult whose age reached their
        retirement age — or who retires via this decision — earns no wages
        this year."""
        adult = state.adult(pid)
        if adult is None or not adult.working:
            return False
        if adult.age >= adult.retirement_age:
            return False
        flag = d.retire_now.adult1 if pid == "a1" else d.retire_now.adult2
        return not flag

    household_wages = sum(a.gross_salary for a in state.adults
                          if works_this_year(a.person_id))

    for pid, attr in (("a1", "k401_a1"), ("a2", "k401_a2")):
        adult = state.adult(pid)
        amount = getattr(c, attr)
        if adult is None:
            setattr(c, attr, clip(f"contributions.401k_{pid}", amount, 0, 0,
                                  "no_such_adult", f"household has no adult {pid}"))
            continue
        if not works_this_year(pid):
            setattr(c, attr, clip(f"contributions.401k_{pid}", amount, 0, 0,
                                  "not_working",
                                  f"{pid} has no wages this year; 401(k) "
                                  "deferrals need compensation"))
            continue
        cap = min(limits.max_401k_employee(adult.age), adult.gross_salary)
        setattr(c, attr, clip(f"contributions.401k_{pid}", amount, 0, cap,
                              "401k_limit",
                              f"401(k) employee deferral capped at {cap:,.0f} for {pid}"))

    # ---- IRA contributions (Roth direct or backdoor) -----------------------
    ira_room_earned = household_wages  # spousal rule: joint earned income covers both
    for pid, attr, flag_attr in (("a1", "roth_ira_a1", "backdoor_roth_a1"),
                                 ("a2", "roth_ira_a2", "backdoor_roth_a2")):
        adult = state.adult(pid)
        amount = getattr(c, attr)
        backdoor = getattr(c, flag_attr)
        if adult is None:
            setattr(c, attr, clip(f"contributions.{attr}", amount, 0, 0,
                                  "no_such_adult", f"household has no adult {pid}"))
            setattr(c, flag_attr, False)
            continue
        cap = limits.max_ira(adult.age)
        if not backdoor:
            frac = limits.roth_ira_allowed_fraction(estimated_magi)
            if frac < 1.0 and amount > cap * frac + 0.01:
                cap_direct = cap * frac
                amount = clip(f"contributions.{attr}", amount, 0, cap_direct,
                              "roth_magi_phaseout",
                              f"direct Roth IRA contribution limited by MAGI "
                              f"~{estimated_magi:,.0f} (use the backdoor flag "
                              f"to contribute the full limit)")
            else:
                amount = clip(f"contributions.{attr}", amount, 0, cap,
                              "ira_limit", f"IRA contribution capped at {cap:,.0f}")
        else:
            amount = clip(f"contributions.{attr}", amount, 0, cap,
                          "ira_limit", f"IRA contribution capped at {cap:,.0f}")
        # earned-income requirement (household level, spousal-friendly)
        if amount > ira_room_earned + 0.01:
            amount = clip(f"contributions.{attr}", amount, 0, max(0.0, ira_room_earned),
                          "ira_earned_income",
                          "IRA contributions require household earned income")
        ira_room_earned = max(0.0, ira_room_earned - amount)
        setattr(c, attr, amount)

    # ---- HSA ---------------------------------------------------------------
    ages = [a.age for a in state.adults]
    if not state.hsa_enabled:
        c.hsa = clip("contributions.hsa", c.hsa, 0, 0, "hsa_not_available",
                     "scenario has no HDHP; HSA contributions not allowed")
    elif max(ages) >= HSA_MAX_AGE:
        c.hsa = clip("contributions.hsa", c.hsa, 0, 0, "hsa_medicare_age",
                     f"HSA contributions end at age {HSA_MAX_AGE} (Medicare)")
    else:
        cap = limits.max_hsa(ages)
        c.hsa = clip("contributions.hsa", c.hsa, 0, cap, "hsa_limit",
                     f"HSA contribution capped at {cap:,.0f}")

    # ---- 529s ---------------------------------------------------------------
    for cid, attr in (("child1", "c529_child1"), ("child2", "c529_child2")):
        amount = getattr(c, attr)
        if state.child(cid) is None or state.account(f"529_{cid}") is None:
            setattr(c, attr, clip(f"contributions.529_{cid}", amount, 0, 0,
                                  "no_such_child", f"household has no {cid}"))
            continue
        setattr(c, attr, clip(f"contributions.529_{cid}", amount, 0,
                              limits.limit_529_per_child, "529_limit",
                              f"529 contribution capped at gift-exclusion proxy "
                              f"{limits.limit_529_per_child:,.0f}"))

    # ---- extra mortgage principal -------------------------------------------
    if state.mortgage is None or not state.mortgage.active:
        c.extra_mortgage_principal = clip(
            "contributions.extra_mortgage_principal", c.extra_mortgage_principal,
            0, 0, "no_mortgage", "no active mortgage to prepay")
    else:
        c.extra_mortgage_principal = clip(
            "contributions.extra_mortgage_principal", c.extra_mortgage_principal,
            0, state.mortgage.balance, "mortgage_overpay",
            "extra principal capped at remaining balance")

    # ---- Roth conversions ----------------------------------------------------
    trad_total = sum(acct.balance for a in state.adults
                     for acct in state.adult_trad_accounts(a.person_id))
    d.roth_conversion_amount = clip(
        "roth_conversion_amount", d.roth_conversion_amount, 0, trad_total,
        "conversion_exceeds_balance",
        f"Roth conversion capped at total pre-tax balances {trad_total:,.0f}")

    # ---- withdrawals -----------------------------------------------------------
    w = d.withdrawals

    def _bal(*account_ids: str) -> float:
        return sum(state.accounts[a].balance for a in account_ids if a in state.accounts)

    w.taxable = clip("withdrawals.taxable", w.taxable, 0, _bal("taxable"),
                     "withdrawal_exceeds_balance", "taxable withdrawal exceeds balance")
    w.trad_401k = clip("withdrawals.trad_401k", w.trad_401k, 0,
                       _bal("trad_401k_a1", "trad_401k_a2"),
                       "withdrawal_exceeds_balance", "401(k) withdrawal exceeds balances")
    w.trad_ira = clip("withdrawals.trad_ira", w.trad_ira, 0,
                      _bal("trad_ira_a1", "trad_ira_a2"),
                      "withdrawal_exceeds_balance", "trad IRA withdrawal exceeds balances")
    w.roth = clip("withdrawals.roth", w.roth, 0, _bal("roth_ira_a1", "roth_ira_a2"),
                  "withdrawal_exceeds_balance", "Roth withdrawal exceeds balances")
    w.hsa = clip("withdrawals.hsa", w.hsa, 0, _bal("hsa"),
                 "withdrawal_exceeds_balance", "HSA withdrawal exceeds balance")

    # ---- allocations ------------------------------------------------------------
    cleaned_allocs: dict[str, dict[str, float]] = {}
    for aid, weights in d.allocations.items():
        if aid not in state.accounts:
            v.append(Violation("unknown_account", f"allocations.{aid}",
                               f"no account '{aid}' in this household"))
            continue
        if aid == "cash":
            if weights and weights != {"cash": 1.0}:
                v.append(Violation("cash_account_alloc", "allocations.cash",
                                   "the cash account is always 100% cash"))
            continue
        if not weights:  # {} → keep prior target
            continue
        wts = {}
        for k in ASSET_CLASSES:
            raw = _clean(weights.get(k, 0.0))
            if math.isnan(raw) or raw < 0:
                if not math.isnan(raw) and raw < -1e-9:
                    v.append(Violation("negative_weight", f"allocations.{aid}.{k}",
                                       "negative allocation weight clipped to 0",
                                       proposed=raw, clipped_to=0.0))
                raw = 0.0
            wts[k] = raw
        extra_keys = set(weights) - set(ASSET_CLASSES)
        if extra_keys:
            v.append(Violation("unknown_asset_class", f"allocations.{aid}",
                               f"unknown asset classes ignored: {sorted(extra_keys)}"))
        total = sum(wts.values())
        if total <= 0:
            v.append(Violation("empty_allocation", f"allocations.{aid}",
                               "allocation weights sum to 0; keeping prior target"))
            continue
        if abs(total - 1.0) > 0.015:
            v.append(Violation("allocation_sum", f"allocations.{aid}",
                               f"weights sum to {total:.3f}; renormalized to 1.0",
                               proposed=total, clipped_to=1.0))
        cleaned_allocs[aid] = {k: val / total for k, val in wts.items()}
    d.allocations = cleaned_allocs

    # ---- retirement / SS claiming flags ------------------------------------------
    for pid, attr in (("a1", "adult1"), ("a2", "adult2")):
        adult = state.adult(pid)
        if getattr(d.retire_now, attr):
            if adult is None:
                setattr(d.retire_now, attr, False)
                v.append(Violation("no_such_adult", f"retire_now.{attr}",
                                   f"household has no adult {pid}"))
            elif adult.retired:
                setattr(d.retire_now, attr, False)
                v.append(Violation("already_retired", f"retire_now.{attr}",
                                   f"{pid} is already retired"))
        if getattr(d.claim_social_security, attr):
            if adult is None:
                setattr(d.claim_social_security, attr, False)
                v.append(Violation("no_such_adult", f"claim_social_security.{attr}",
                                   f"household has no adult {pid}"))
            elif adult.ss_claimed:
                setattr(d.claim_social_security, attr, False)
                v.append(Violation("already_claimed", f"claim_social_security.{attr}",
                                   f"{pid} already claimed Social Security"))
            elif adult.age < SS_EARLIEST_CLAIM_AGE:
                setattr(d.claim_social_security, attr, False)
                v.append(Violation("ss_too_young", f"claim_social_security.{attr}",
                                   f"{pid} is {adult.age}; earliest claim age is "
                                   f"{SS_EARLIEST_CLAIM_AGE}"))

    # ---- tax-loss harvesting: only meaningful when there is a loss ---------------
    taxable = state.account("taxable")
    if d.tax_loss_harvest and (taxable is None or taxable.basis <= taxable.balance + 0.01):
        d.tax_loss_harvest = False
        v.append(Violation("no_loss_to_harvest", "tax_loss_harvest",
                           "taxable account has no unrealized loss; flag ignored"))

    return d, v
