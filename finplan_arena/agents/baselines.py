"""Rule-based baseline agents.

Four personas an LLM should be measured against:

* ``drift``   — absolute passivity: no contributions, no allocation changes,
                spending pinned at the floor; the engine's forced cascade,
                RMD enforcement, and age-70 SS auto-claim do everything.
* ``tdf``     — target-date-fund saver: (110 − age) glide path, flat 15%
                gross savings rate, straight-line 529 funding, claims SS at
                FRA (67), 4% rule in retirement.
* ``naive``   — 60/40 everywhere, spends first and sweeps what's left into a
                taxable account, no tax planning at all (forgoes the employer
                match — deliberately), claims SS at 62.
* ``expert``  — the bar: TDF glide path + asset location (bonds in
                tax-deferred first), contribution priority
                match → HSA → Roth/backdoor → 401(k) → 529 → taxable,
                Roth conversions filling the 12% bracket in early retirement,
                delays SS to 70 for the higher earner, harvests losses in
                down years.

All baselines are deterministic and are expected to produce **zero**
validation violations (tested) — they read the same observation an LLM gets
and respect the same limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.accounts import (SS_FULL_RETIREMENT_AGE, ContributionLimits,
                             ss_claiming_factor)
from ..core.state import HouseholdState
from ..core.tax_engine import (TaxParams, TaxYearInput,
                               ordinary_room_to_bracket_top)
from .base import BaseAgent, Decision, Observation

STOCK_SPLIT_US = 0.70  # of the equity sleeve: 70% US / 30% international

#: crude effective-tax haircuts used only for *planning* budgets inside
#: baseline heuristics (the engine computes real taxes; these just keep
#: baseline cash budgets from overcommitting). The expert plans a little
#: sharper — closer to the true all-in average rate at these incomes.
NET_OF_TAX = 0.72
NET_OF_TAX_EXPERT = 0.78


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _st(obs: Observation) -> HouseholdState:
    return HouseholdState.from_dict(obs.state)


def _limits(st: HouseholdState) -> ContributionLimits:
    return ContributionLimits.for_year(st.inflation_index, st.filing_status)


def _glide_equity(avg_age: float, lo: float = 0.30, hi: float = 0.90) -> float:
    return min(hi, max(lo, (110.0 - avg_age) / 100.0))


def _equity_alloc(equity: float, cash_frac: float = 0.0) -> dict[str, float]:
    bonds = max(0.0, 1.0 - equity - cash_frac)
    return {"stocks_us": equity * STOCK_SPLIT_US,
            "stocks_intl": equity * (1 - STOCK_SPLIT_US),
            "bonds": bonds, "cash": cash_frac}


def _alloc_529(child_age: int, college_start: int) -> dict[str, float]:
    years_out = max(0, college_start - child_age)
    equity = min(0.80, max(0.10, 0.08 * years_out))
    remainder = 1.0 - equity
    return {"stocks_us": equity * STOCK_SPLIT_US,
            "stocks_intl": equity * (1 - STOCK_SPLIT_US),
            "bonds": remainder * 0.75, "cash": remainder * 0.25}


def _mortgage_annual(st: HouseholdState) -> float:
    if st.mortgage is None or not st.mortgage.active:
        return 0.0
    return min(12 * st.mortgage.monthly_payment,
               st.mortgage.balance * (1 + st.mortgage.annual_rate))


def _working_this_year(adult) -> bool:
    """Will this adult actually earn wages this sim year? The engine retires
    anyone whose age reached their retirement_age at the START of the year,
    so agents must anticipate it (the observed ``retired`` flag lags one
    transition)."""
    return adult.working and adult.age < adult.retirement_age


def _gross_income(st: HouseholdState) -> float:
    return sum(a.gross_salary for a in st.adults if _working_this_year(a))


def _ss_income(st: HouseholdState) -> float:
    return sum(a.ss_annual_benefit for a in st.adults if a.ss_claimed)


def _all_retired(st: HouseholdState) -> bool:
    return all(not _working_this_year(a) for a in st.adults)


def _college_bills(st: HouseholdState) -> float:
    return sum(c.college_annual_cost for c in st.children if c.in_college())


def _clamp_disc(st: HouseholdState, x: float) -> float:
    return min(max(x, st.discretionary_floor), st.discretionary_ceiling)


def _cash_buffer(st: HouseholdState) -> float:
    """Emergency fund target: six months of essential + mortgage outflow."""
    return 0.5 * (st.essential_expenses + _mortgage_annual(st))


def _sweep_excess_cash(st: HouseholdState, committed: float = 0.0) -> float:
    """Start-of-year cash above the buffer (and above ``committed`` spending
    already counting on it) that can safely move into taxable."""
    cash = st.accounts["cash"].balance if st.account("cash") else 0.0
    return max(0.0, cash - _cash_buffer(st) - committed)


def _straight_line_529(st: HouseholdState, child, fraction: float,
                       limit: float) -> float:
    """Contribution that funds ``fraction`` of remaining college cost on a
    straight line by college start (0 once college begins)."""
    acct = st.account(f"529_{child.person_id}")
    if acct is None or fraction <= 0:
        return 0.0
    years_out = child.college_start_age - child.age
    if years_out <= 0:
        return 0.0
    target = fraction * child.college_years * child.college_annual_cost
    needed = max(0.0, target - acct.balance)
    return min(needed / years_out, limit)


def _split_withdrawal(st: HouseholdState, needed: float,
                      order: tuple[str, ...] = ("taxable", "trad", "roth"),
                      ) -> dict[str, float]:
    """Plan withdrawals covering ``needed``, in the given source order,
    capped by balances. Returns decision-field amounts."""
    out = {"taxable": 0.0, "trad_401k": 0.0, "trad_ira": 0.0, "roth": 0.0}
    remaining = max(0.0, needed)
    for src in order:
        if remaining <= 0:
            break
        if src == "taxable":
            bal = st.accounts["taxable"].balance if st.account("taxable") else 0.0
            take = min(remaining, bal)
            out["taxable"] = take
            remaining -= take
        elif src == "trad":
            bal_401k = sum(st.accounts[a].balance for a in
                           ("trad_401k_a1", "trad_401k_a2") if a in st.accounts)
            take = min(remaining, bal_401k)
            out["trad_401k"] = take
            remaining -= take
            bal_ira = sum(st.accounts[a].balance for a in
                          ("trad_ira_a1", "trad_ira_a2") if a in st.accounts)
            take = min(remaining, bal_ira)
            out["trad_ira"] = take
            remaining -= take
        elif src == "roth":
            bal = sum(st.accounts[a].balance for a in
                      ("roth_ira_a1", "roth_ira_a2") if a in st.accounts)
            take = min(remaining, bal)
            out["roth"] = take
            remaining -= take
    return out


def _sustainable_spending(st: HouseholdState,
                          planned_claim_ages: dict[str, int]) -> float:
    """Bridge-aware sustainable total spending (nominal $/yr).

    Delaying Social Security only works with a *bridge*: reserve the
    portfolio dollars that will replace the unclaimed benefits until each
    claim date, apply the 4%-rule to what remains, and count the expected
    benefits as if already flowing. Without this, a delayer starves from
    retirement to claim age in bear markets — the classic mistake."""
    investable = sum(a.balance for aid, a in st.accounts.items()
                     if not aid.startswith("529"))
    expected_ss = 0.0
    bridge_reserve = 0.0
    for a in st.adults:
        if a.ss_claimed:
            expected_ss += a.ss_annual_benefit
            continue
        claim_age = min(70, max(planned_claim_ages.get(a.person_id, 70), a.age))
        benefit = a.ss_pia_annual * ss_claiming_factor(claim_age)
        expected_ss += benefit
        bridge_reserve += benefit * max(0, claim_age - a.age)
    return 0.04 * max(0.0, investable - bridge_reserve) + expected_ss


@dataclass
class _FourPercentRule:
    """Tracks the classic 4% rule: 4% of the portfolio at retirement,
    inflation-indexed thereafter."""

    initial_nominal: float = 0.0
    index_at_start: float = 1.0
    started: bool = False

    def annual_amount(self, st: HouseholdState, rate: float = 0.04) -> float:
        if not self.started:
            investable = sum(a.balance for aid, a in st.accounts.items()
                             if not aid.startswith("529"))
            self.initial_nominal = rate * investable
            self.index_at_start = st.inflation_index
            self.started = True
        return self.initial_nominal * (st.inflation_index / self.index_at_start)


# ---------------------------------------------------------------------------
# 1. Target-date-fund baseline
# ---------------------------------------------------------------------------


class TargetDateAgent(BaseAgent):
    name = "tdf"

    def __init__(self) -> None:
        self._rule = _FourPercentRule()

    def reset(self) -> None:
        self._rule = _FourPercentRule()

    def decide(self, obs: Observation) -> Decision:
        st = _st(obs)
        limits = _limits(st)
        avg_age = sum(a.age for a in st.adults) / len(st.adults)
        equity = _glide_equity(avg_age, lo=0.25, hi=0.90)

        allocations = {}
        for aid, acct in st.accounts.items():
            if aid == "cash":
                continue
            if aid.startswith("529"):
                child = st.child(acct.owner) if acct.owner else None
                if child:
                    allocations[aid] = _alloc_529(child.age, child.college_start_age)
            else:
                allocations[aid] = _equity_alloc(equity)

        d = Decision(rationale="TDF baseline: glide path, 15% savings, "
                               "straight-line 529, SS at 67, 4% rule.",
                     allocations=allocations)

        gross = _gross_income(st)
        mortgage = _mortgage_annual(st)

        if not _all_retired(st):
            # --- accumulation: save 15% of gross, 401(k) first ------------
            budget = 0.15 * gross
            for adult, attr in ((st.adult("a1"), "k401_a1"),
                                (st.adult("a2"), "k401_a2")):
                if adult is None or not _working_this_year(adult) or budget <= 0:
                    continue
                cap = min(limits.max_401k_employee(adult.age), adult.gross_salary)
                amt = min(0.15 * adult.gross_salary, cap, budget)
                setattr(d.contributions, attr, round(amt, 2))
                budget -= amt
            # 529s on a straight line
            for child in st.children:
                amt = _straight_line_529(st, child, st.goals.college_fund_fraction,
                                         limits.limit_529_per_child)
                if child.person_id == "child1":
                    d.contributions.c529_child1 = round(amt, 2)
                else:
                    d.contributions.c529_child2 = round(amt, 2)
            # leftover of the 15% (if 401k capped) goes to taxable,
            # plus any idle cash above the emergency buffer
            sweep = _sweep_excess_cash(st)
            if budget + sweep > 1:
                d.contributions.taxable = round(max(0.0, budget) + sweep, 2)
            disposable = (NET_OF_TAX * gross - st.essential_expenses - mortgage
                          - 0.15 * gross)
            d.annual_spending_discretionary = round(_clamp_disc(st, disposable), 2)
        else:
            # --- retirement: 4% rule, SS at FRA ----------------------------
            spend = self._rule.annual_amount(st)
            ss = _ss_income(st)
            total_target = spend + ss
            disc = total_target - st.essential_expenses - mortgage - _college_bills(st)
            d.annual_spending_discretionary = round(_clamp_disc(st, disc), 2)
            cash_bal = st.accounts["cash"].balance
            committed = max(0.0, total_target * 1.15 - ss)
            sweep = _sweep_excess_cash(st, committed=committed)
            if sweep > 1:
                d.contributions.taxable = round(sweep, 2)
            needed = max(0.0, committed - cash_bal)  # 15% tax cushion
            w = _split_withdrawal(st, needed)
            d.withdrawals.taxable = round(w["taxable"], 2)
            d.withdrawals.trad_401k = round(w["trad_401k"], 2)
            d.withdrawals.trad_ira = round(w["trad_ira"], 2)
            d.withdrawals.roth = round(w["roth"], 2)

        for adult, attr in ((st.adult("a1"), "adult1"), (st.adult("a2"), "adult2")):
            if (adult is not None and not adult.ss_claimed
                    and adult.age >= SS_FULL_RETIREMENT_AGE):
                setattr(d.claim_social_security, attr, True)
        return d


# ---------------------------------------------------------------------------
# 2. Naive baseline
# ---------------------------------------------------------------------------


class NaiveAgent(BaseAgent):
    """Spends first, saves what's left in a taxable account, no tax planning
    (not even the employer match), 60/40 everywhere, claims SS at 62."""

    name = "naive"

    _ALLOC_6040 = {"stocks_us": 0.45, "stocks_intl": 0.15,
                   "bonds": 0.35, "cash": 0.05}

    def decide(self, obs: Observation) -> Decision:
        st = _st(obs)
        allocations = {aid: dict(self._ALLOC_6040)
                       for aid in st.accounts if aid != "cash"}
        d = Decision(rationale="Naive baseline: 60/40, spend first, sweep "
                               "leftovers to taxable, claim SS at 62.",
                     allocations=allocations)

        gross = _gross_income(st)
        mortgage = _mortgage_annual(st)

        if not _all_retired(st):
            disc = _clamp_disc(st, 0.62 * gross - st.essential_expenses - mortgage)
            d.annual_spending_discretionary = round(disc, 2)
            sweep = max(0.0, 0.62 * gross - st.essential_expenses - mortgage - disc)
            leftover = max(0.0, NET_OF_TAX * gross - st.essential_expenses
                           - mortgage - disc)
            d.contributions.taxable = round(min(leftover, max(sweep, leftover)), 2)
        else:
            target = st.goals.retirement_spending_real * st.inflation_index
            disc = _clamp_disc(st, target - st.essential_expenses)
            d.annual_spending_discretionary = round(disc, 2)
            # no withdrawal planning: the forced cascade pays the bills

        for adult, attr in ((st.adult("a1"), "adult1"), (st.adult("a2"), "adult2")):
            if adult is not None and not adult.ss_claimed and adult.age >= 62:
                setattr(d.claim_social_security, attr, True)
        return d


# ---------------------------------------------------------------------------
# 3. Rule-based expert baseline
# ---------------------------------------------------------------------------


class RuleBasedExpertAgent(BaseAgent):
    """The bar an LLM should beat or match. See module docstring."""

    name = "expert"

    def decide(self, obs: Observation) -> Decision:
        st = _st(obs)
        limits = _limits(st)
        d = Decision(rationale="Expert baseline: glide path with asset "
                               "location, match→HSA→Roth→401k→529→taxable, "
                               "12%-bracket Roth conversions in early "
                               "retirement, SS at 70 for the higher earner, "
                               "TLH in down years.")

        d.allocations = self._asset_location(st)

        if not _all_retired(st):
            self._accumulate(st, limits, d)
        else:
            self._decumulate(st, limits, d)

        # --- Social Security: higher earner waits to 70, other claims at FRA
        adults = sorted(st.adults, key=lambda a: -a.ss_pia_annual)
        for i, adult in enumerate(adults):
            if adult.ss_claimed:
                continue
            claim_at = 70 if i == 0 else SS_FULL_RETIREMENT_AGE
            if adult.age >= claim_at:
                attr = "adult1" if adult.person_id == "a1" else "adult2"
                setattr(d.claim_social_security, attr, True)

        # --- tax-loss harvest whenever a loss exists ------------------------
        taxable = st.account("taxable")
        if taxable is not None and taxable.basis > taxable.balance + 1:
            d.tax_loss_harvest = True
        return d

    # -- allocation with asset location ----------------------------------

    def _asset_location(self, st: HouseholdState) -> dict[str, dict[str, float]]:
        avg_age = sum(a.age for a in st.adults) / len(st.adults)
        equity_frac = _glide_equity(avg_age, lo=0.35, hi=0.90)
        housed = {aid: a for aid, a in st.accounts.items()
                  if aid != "cash" and not aid.startswith("529")}
        total = sum(a.balance for a in housed.values())
        allocations: dict[str, dict[str, float]] = {}
        if total > 0:
            bond_budget = (1.0 - equity_frac) * total
            # bonds live in pre-tax accounts first, then taxable; Roth/HSA
            # hold equities (highest expected growth in tax-free space)
            order = [aid for aid in ("trad_401k_a1", "trad_401k_a2",
                                     "trad_ira_a1", "trad_ira_a2", "taxable")
                     if aid in housed]
            bond_assign = {aid: 0.0 for aid in housed}
            for aid in order:
                if bond_budget <= 0:
                    break
                take = min(housed[aid].balance, bond_budget)
                bond_assign[aid] = take
                bond_budget -= take
            for aid, acct in housed.items():
                if acct.balance <= 0:
                    allocations[aid] = _equity_alloc(equity_frac)
                    continue
                bond_frac = min(1.0, bond_assign[aid] / acct.balance)
                eq = 1.0 - bond_frac
                allocations[aid] = {"stocks_us": eq * STOCK_SPLIT_US,
                                    "stocks_intl": eq * (1 - STOCK_SPLIT_US),
                                    "bonds": bond_frac, "cash": 0.0}
        else:
            for aid in housed:
                allocations[aid] = _equity_alloc(equity_frac)
        for child in st.children:
            aid = f"529_{child.person_id}"
            if aid in st.accounts:
                allocations[aid] = _alloc_529(child.age, child.college_start_age)
        return allocations

    # -- lifecycle savings requirement ------------------------------------

    def _expected_ss_real(self, st: HouseholdState) -> float:
        """Household SS at the expert's planned claim ages (70 for the
        higher earner, FRA for the other), in today's real dollars."""
        adults = sorted(st.adults, key=lambda a: -a.ss_pia_annual)
        total = 0.0
        for i, a in enumerate(adults):
            factor = 1.24 if i == 0 else 1.0
            total += a.ss_pia_annual * factor
        return total / st.inflation_index

    def _required_annual_saving(self, st: HouseholdState) -> float:
        """Retirement-goal savings requirement (nominal $/yr), amortized
        straight-line to the last retirement date.

        Nest egg needed ≈ safety × 25 × (target spending − haircut·SS);
        the 20% SS haircut and 15% safety margin are prudence, the 25×
        multiplier is the 4%-rule inverse. Ignoring future real growth on
        assets is a further conservative bias."""
        target = st.goals.retirement_spending_real
        need_real = 1.15 * 25.0 * max(0.0, target - 0.80 * self._expected_ss_real(st))
        assets_real = sum(a.balance for aid, a in st.accounts.items()
                          if not aid.startswith("529") and aid != "cash"
                          ) / st.inflation_index
        gap_real = max(0.0, need_real - assets_real)
        years = [a.retirement_age - a.age for a in st.adults
                 if _working_this_year(a)]
        horizon = max(1, max(years) if years else 1)
        return (gap_real / horizon) * st.inflation_index

    # -- working years ------------------------------------------------------

    def _accumulate(self, st: HouseholdState, limits: ContributionLimits,
                    d: Decision) -> None:
        gross = _gross_income(st)
        mortgage = _mortgage_annual(st)

        # match capture is non-negotiable (free money); beyond that, save
        # what the retirement goal actually requires — the CE objective
        # punishes dying rich as much as dying broke
        match_floor = sum(a.match_cap_fraction * a.gross_salary
                          for a in st.adults if _working_this_year(a))
        required = self._required_annual_saving(st)
        budget = min(max(match_floor, required), 0.35 * gross)

        # (1) capture the full employer match
        for adult, attr in ((st.adult("a1"), "k401_a1"), (st.adult("a2"), "k401_a2")):
            if adult is None or not _working_this_year(adult):
                continue
            amt = min(adult.match_cap_fraction * adult.gross_salary,
                      limits.max_401k_employee(adult.age), max(0.0, budget))
            setattr(d.contributions, attr, round(max(0.0, amt), 2))
            budget -= amt

        # (2) HSA to the max (triple tax advantage beats everything after
        #     the match)
        if st.hsa_enabled and budget > 0:
            ages = [a.age for a in st.adults]
            if max(ages) < 65:
                amt = min(limits.max_hsa(ages), budget)
                d.contributions.hsa = round(amt, 2)
                budget -= amt

        # (3) Roth IRAs (backdoor when the front door is phased out)
        est_magi = max(0.0, gross - d.contributions.k401_a1
                       - d.contributions.k401_a2 - d.contributions.hsa)
        use_backdoor = est_magi >= limits.roth_magi_phaseout_lo
        for adult, amt_attr, flag_attr in (
                (st.adult("a1"), "roth_ira_a1", "backdoor_roth_a1"),
                (st.adult("a2"), "roth_ira_a2", "backdoor_roth_a2")):
            if adult is None or budget <= 0:
                continue
            amt = min(limits.max_ira(adult.age), budget)
            if amt <= 0:
                continue
            setattr(d.contributions, amt_attr, round(amt, 2))
            if use_backdoor:
                setattr(d.contributions, flag_attr, True)
            budget -= amt

        # (4) 401(k)s to the employee max
        for adult, attr in ((st.adult("a1"), "k401_a1"), (st.adult("a2"), "k401_a2")):
            if adult is None or not _working_this_year(adult) or budget <= 0:
                continue
            current = getattr(d.contributions, attr)
            cap = min(limits.max_401k_employee(adult.age), adult.gross_salary)
            extra = min(cap - current, budget)
            if extra > 0:
                setattr(d.contributions, attr, round(current + extra, 2))
                budget -= extra

        # (5) 529s on the straight line (outside the savings budget —
        #     college is a scheduled bill, not retirement savings)
        c529_total = 0.0
        for child in st.children:
            amt = _straight_line_529(st, child, st.goals.college_fund_fraction,
                                     limits.limit_529_per_child)
            c529_total += amt
            if child.person_id == "child1":
                d.contributions.c529_child1 = round(amt, 2)
            else:
                d.contributions.c529_child2 = round(amt, 2)

        # (6) leftovers of the budget go to taxable, plus idle cash above
        #     the emergency buffer (the sweep is a stock transfer, so it is
        #     excluded from the consumption budget below)
        sweep = _sweep_excess_cash(st)
        if budget + sweep > 1:
            d.contributions.taxable = round(max(0.0, budget) + sweep, 2)

        pretax = d.contributions.k401_a1 + d.contributions.k401_a2 + d.contributions.hsa
        after_tax_outlay = (d.contributions.roth_ira_a1 + d.contributions.roth_ira_a2
                            + max(0.0, budget) + c529_total)
        college_oop = max(0.0, _college_bills(st)
                          - sum(st.accounts[a].balance for a in st.accounts
                                if a.startswith("529")))
        disposable = (NET_OF_TAX_EXPERT * (gross - pretax) + _ss_income(st)
                      - st.essential_expenses - mortgage - after_tax_outlay
                      - college_oop)
        d.annual_spending_discretionary = round(_clamp_disc(st, disposable), 2)

    # -- retirement ----------------------------------------------------------

    def _decumulate(self, st: HouseholdState, limits: ContributionLimits,
                    d: Decision) -> None:
        mortgage = _mortgage_annual(st)
        ss = _ss_income(st)
        target_total = st.goals.retirement_spending_real * st.inflation_index
        # spend the larger of the goal and the bridge-aware sustainable draw —
        # wealth never consumed (or bequeathed at tiny marginal utility) is
        # wasted under the CE objective, and SS delay must be bridged
        adults_by_pia = sorted(st.adults, key=lambda a: -a.ss_pia_annual)
        planned = {a.person_id: (70 if i == 0 else SS_FULL_RETIREMENT_AGE)
                   for i, a in enumerate(adults_by_pia)}
        sustainable_total = _sustainable_spending(st, planned)
        spend_total = max(target_total, sustainable_total)
        disc = _clamp_disc(st, spend_total - st.essential_expenses - mortgage
                           - _college_bills(st))
        d.annual_spending_discretionary = round(disc, 2)

        spend_needed = (st.essential_expenses + mortgage + disc
                        + _college_bills(st))
        cash_bal = st.accounts["cash"].balance
        sweep = _sweep_excess_cash(st, committed=max(0.0, spend_needed * 1.12 - ss))
        if sweep > 1:
            d.contributions.taxable = round(sweep, 2)

        params = TaxParams.for_year(st.filing_status, st.inflation_index)
        taxable_acct = st.account("taxable")
        interest_est = 0.0
        if taxable_acct and taxable_acct.balance > 0:
            interest_est = (taxable_acct.balance
                            * (taxable_acct.alloc.equity_fraction() * 0.02
                               + taxable_acct.alloc.bonds * 0.035))

        # withdrawals: taxable first; trad only up to the 12% bracket top;
        # Roth for the rest (keeps ordinary income cheap)
        needed = max(0.0, spend_needed * 1.12 - ss - cash_bal)
        w = _split_withdrawal(st, needed, order=("taxable",))
        still_needed = needed - w["taxable"]
        trad_room = ordinary_room_to_bracket_top(
            TaxYearInput(interest_ordinary=interest_est, ss_benefits_gross=ss),
            params, 0.12)
        if still_needed > 0:
            trad_bal = sum(st.accounts[a].balance for a in
                           ("trad_401k_a1", "trad_401k_a2",
                            "trad_ira_a1", "trad_ira_a2") if a in st.accounts)
            all_penalty_free = all(a.age >= 60 for a in st.adults)
            trad_cap = trad_room if not all_penalty_free else max(trad_room,
                                                                  still_needed)
            take = min(still_needed, trad_bal, max(0.0, trad_cap))
            w401 = min(take, sum(st.accounts[a].balance for a in
                                 ("trad_401k_a1", "trad_401k_a2")
                                 if a in st.accounts))
            w["trad_401k"] = w401
            w["trad_ira"] = take - w401
            still_needed -= take
        if still_needed > 0:
            roth_bal = sum(st.accounts[a].balance for a in
                           ("roth_ira_a1", "roth_ira_a2") if a in st.accounts)
            w["roth"] = min(still_needed, roth_bal)
        d.withdrawals.taxable = round(w["taxable"], 2)
        d.withdrawals.trad_401k = round(w.get("trad_401k", 0.0), 2)
        d.withdrawals.trad_ira = round(w.get("trad_ira", 0.0), 2)
        d.withdrawals.roth = round(w.get("roth", 0.0), 2)

        # Roth conversions: fill what's left of the 12% bracket before SS/RMDs
        any_rmd_age = any(a.age >= 72 for a in st.adults)
        if not any_rmd_age:
            trad_wd_planned = d.withdrawals.trad_401k + d.withdrawals.trad_ira
            room = ordinary_room_to_bracket_top(
                TaxYearInput(interest_ordinary=interest_est,
                             trad_taxable_withdrawals=trad_wd_planned,
                             ss_benefits_gross=ss),
                params, 0.12)
            trad_bal = sum(st.accounts[a].balance for a in
                           ("trad_401k_a1", "trad_401k_a2",
                            "trad_ira_a1", "trad_ira_a2") if a in st.accounts)
            convert = min(max(0.0, room), max(0.0, trad_bal - trad_wd_planned))
            if convert > 500:
                d.roth_conversion_amount = round(convert, 2)


# ---------------------------------------------------------------------------
# Passive drawdown policy (post-active-years autopilot)
# ---------------------------------------------------------------------------


class FourPercentDrawdownAgent(BaseAgent):
    """Autopilot used after the agent's active window ends (spec §8): keep
    the final allocations, spend by the 4% rule (of the portfolio at
    handoff, inflation-indexed), withdraw taxable → pre-tax → Roth, claim
    any unclaimed Social Security at 70 (the engine forces this anyway),
    and let RMD enforcement do its thing."""

    name = "drawdown-4pct"

    def __init__(self) -> None:
        self._rule = _FourPercentRule()

    def reset(self) -> None:
        self._rule = _FourPercentRule()

    def decide(self, obs: Observation) -> Decision:
        st = _st(obs)
        d = Decision(rationale="passive drawdown: hold allocations, 4% rule")
        mortgage = _mortgage_annual(st)
        ss = _ss_income(st)
        spend = self._rule.annual_amount(st)
        # bridge-aware floor: if SS is still being delayed (claims are forced
        # at 70), spend against the post-claim income picture rather than
        # starving through the bridge years
        sustainable = _sustainable_spending(st, {a.person_id: 70
                                                 for a in st.adults})
        total_spend = max(spend + ss, sustainable)
        disc = _clamp_disc(st, total_spend - st.essential_expenses - mortgage
                           - _college_bills(st))
        d.annual_spending_discretionary = round(disc, 2)
        total_needed = (st.essential_expenses + mortgage + disc
                        + _college_bills(st))
        cash_bal = st.accounts["cash"].balance
        needed = max(0.0, total_needed * 1.12 - ss - cash_bal)
        w = _split_withdrawal(st, needed)
        d.withdrawals.taxable = round(w["taxable"], 2)
        d.withdrawals.trad_401k = round(w["trad_401k"], 2)
        d.withdrawals.trad_ira = round(w["trad_ira"], 2)
        d.withdrawals.roth = round(w["roth"], 2)
        # allocations: leave empty → every account keeps its prior target
        return d


class DriftAgent(BaseAgent):
    """Absolute passivity: the household never lifts a finger.

    No contributions (even the employer match is forgone), no voluntary
    withdrawals, no conversions, no harvesting, no claims, and allocations
    are never touched (every account keeps its scenario-initial targets).
    Salary piles up in cash; in retirement the forced-liquidation cascade
    pays the bills, RMDs are engine-enforced, and Social Security arrives
    only via the engine's forced claim at 70. The single non-default field
    is discretionary spending, pinned at the scenario floor so the agent
    stays zero-violation (a literal 0 would just be clipped to the floor
    and flagged every year).

    This is the true bottom anchor of the ladder: an LLM's lift over
    ``drift`` measures the value of doing anything at all."""

    name = "drift"

    def decide(self, obs: Observation) -> Decision:
        st = _st(obs)
        return Decision(
            rationale="Drift baseline: touch nothing; spend the floor; the "
                      "engine's forced cascade, RMDs, and age-70 SS claim "
                      "do the rest.",
            annual_spending_discretionary=round(st.discretionary_floor, 2))


BASELINES: dict[str, type[BaseAgent]] = {
    "tdf": TargetDateAgent,
    "naive": NaiveAgent,
    "expert": RuleBasedExpertAgent,
    "drift": DriftAgent,
}
