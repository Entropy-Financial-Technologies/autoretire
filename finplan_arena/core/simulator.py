"""The annual step function.

``simulate_year(state, decision, year_returns, cfg)`` is **deterministic**:
given the same state, decision, and returns it produces the identical new
state and log. All randomness lives in the return generator.

Order of operations inside one sim year (mirrors the eval spec):

1.  Life events at year start: planned/decided retirements, SS claims
    (incl. the forced claim at 70), college window transitions.
2.  Income: salaries for working adults, SS benefits for claimants.
3.  Agent decision applied after validation/clipping (rule feasibility was
    checked by the caller via ``validation.validate_decision``; resource
    feasibility — actual cash — is enforced here, recording violations).
    Includes payroll deferrals + employer match, voluntary withdrawals,
    RMD enforcement, Roth conversions, after-tax contributions, tax-loss
    harvesting, and rebalancing to target allocations.
4.  Market returns applied per asset class per account; taxable dividends /
    interest recognized (modeled as reinvested → they add to basis).
5.  Taxes settled (settlement #1): tax on everything realized so far is paid
    from cash; if cash is short, the forced-liquidation cascade raises it
    (each pull adds its own tax; iterate to convergence).
6.  Expenses paid in priority order — essential → mortgage → college
    (own 529 first, qualified) → discretionary — with the shortfall cascade:
    cash → taxable → Roth basis → pre-tax (10% penalty if under 59.5; the
    trad-IRA higher-education exception applies to the college tier) →
    HSA → Roth earnings → non-qualified 529. Unmet essential/mortgage/
    college amounts are recorded as shortfall events.
7.  Tax true-up (settlement #2) on income the cascade realized.
8.  Advance: ages +1, salary growth, CPI indexation of expenses/limits/
    brackets, COLA on SS, college-cost inflation at CPI + premium, logging.

Taxes are paid in the year the income arises (continuous withholding — a
documented simplification; there is no April settlement lag).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..agents.base import Decision, Observation
from ..agents.validation import Violation, validate_decision
from ..markets.returns import YearReturns
from . import life_events as life
from .accounts import (PENALTY_FREE_AGE, ContributionLimits, employer_match,
                       backdoor_roth_contribution, contribute_to_account,
                       household_rmd_requirements, roth_convert,
                       withdraw_from_account)
from .state import Account, AccountKind, Allocation, HouseholdState
from .tax_engine import TaxParams, TaxYearInput, compute_taxes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Static engine parameters (identical across years and agents)."""

    dividend_yield: float = 0.02        # qualified dividend yield on equities (taxable)
    bond_income_yield: float = 0.035    # ordinary-income yield on bonds (taxable)
    college_inflation_premium: float = 0.025  # college inflates at CPI + this
    enable_ctc: bool = True
    enable_niit: bool = True
    settle_iterations: int = 15         # forced-liquidation fixed-point cap


# ---------------------------------------------------------------------------
# Per-year log
# ---------------------------------------------------------------------------


@dataclass
class YearLog:
    year_index: int = 0
    calendar_year: int = 0
    returns: dict = field(default_factory=dict)
    ages: dict = field(default_factory=dict)
    inflation_index_boy: float = 1.0    # start-of-year cumulative CPI factor
    all_retired: bool = False

    decision: dict = field(default_factory=dict)          # validated, by alias
    rationale: str = ""
    violations: list = field(default_factory=list)        # Violation dicts
    life_events: dict = field(default_factory=dict)

    salaries: dict = field(default_factory=dict)
    employer_match_total: float = 0.0
    ss_benefits: float = 0.0

    contributions_applied: dict = field(default_factory=dict)
    withdrawals_voluntary: dict = field(default_factory=dict)
    rmd_forced: float = 0.0
    roth_converted: float = 0.0
    forced_liquidation: float = 0.0     # gross raised by cascades (tax + expenses)

    tax: dict = field(default_factory=dict)
    unpaid_tax: float = 0.0

    expenses: dict = field(default_factory=dict)
    shortfalls: list = field(default_factory=list)        # [{tier, amount}]

    allocations_target: dict = field(default_factory=dict)
    eoy_balances: dict = field(default_factory=dict)
    mortgage_balance: float = 0.0
    net_worth_nominal: float = 0.0
    net_worth_real: float = 0.0
    inflation_index_eoy: float = 1.0

    consumption_nominal: float = 0.0    # essential_paid + discretionary_paid
    consumption_real: float = 0.0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "YearLog":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ---------------------------------------------------------------------------
# Cash plumbing
# ---------------------------------------------------------------------------

#: Forced-liquidation cascade order (spec §3.6). Tranche names are consumed
#: by ``_raise_cash``. Roth is split into a basis tranche (tax/penalty-free)
#: and a last-resort earnings tranche.
CASCADE = ("taxable", "roth_basis", "pretax", "hsa", "roth_earnings", "529")


def _cash(state: HouseholdState) -> Account:
    return state.accounts["cash"]


def _take_cash(state: HouseholdState, amount: float) -> float:
    c = _cash(state)
    take = min(max(0.0, amount), c.balance)
    c.balance -= take
    return take


def _adult_age(state: HouseholdState, person_id: Optional[str]) -> int:
    a = state.adult(person_id) if person_id else None
    return a.age if a else 99


def _raise_cash(state: HouseholdState, needed: float, T: TaxYearInput,
                tier: str) -> float:
    """Forced-liquidation cascade: pull ``needed`` gross dollars into cash.

    Follows CASCADE order. Adds tax character to ``T`` as it goes (the caller
    re-runs tax settlement afterwards). Prefers penalty-free owners within
    the pre-tax tranche. Returns gross raised (≤ needed).
    """
    raised = 0.0

    def pull(acct: Account, amount: float, *, owner_age: int = 99,
             qualified_529: bool = False, penalty_exempt: bool = False,
             cap: Optional[float] = None) -> float:
        amt = min(amount, cap) if cap is not None else amount
        if amt <= 0 or acct.balance <= 0:
            return 0.0
        eff = withdraw_from_account(acct, amt, owner_age,
                                    qualified_529=qualified_529,
                                    penalty_exempt=penalty_exempt)
        T.trad_taxable_withdrawals += (eff.ordinary_income
                                       if acct.kind in (AccountKind.TRAD_401K,
                                                        AccountKind.TRAD_IRA) else 0.0)
        if acct.kind == AccountKind.ROTH_IRA and eff.ordinary_income > 0:
            T.other_ordinary_income += eff.ordinary_income  # young Roth earnings
        if acct.kind == AccountKind.COLLEGE_529 and eff.ordinary_income > 0:
            T.other_ordinary_income += eff.ordinary_income  # non-qualified 529 earnings
        T.realized_lt_gains += eff.realized_lt_gain
        T.penalty_base += eff.penalty_base
        _cash(state).balance += eff.gross
        return eff.gross

    for tranche in CASCADE:
        if raised >= needed - 1e-9:
            break
        remaining = needed - raised
        if tranche == "taxable":
            acct = state.account("taxable")
            if acct:
                raised += pull(acct, remaining)
        elif tranche == "roth_basis":
            for adult in state.adults:
                acct = state.account(f"roth_ira_{adult.person_id}")
                if acct and acct.basis > 0:
                    raised += pull(acct, needed - raised, owner_age=adult.age,
                                   cap=acct.basis)
                if raised >= needed - 1e-9:
                    break
        elif tranche == "pretax":
            # penalty-free owners first
            pairs = []
            for adult in state.adults:
                for acct in state.adult_trad_accounts(adult.person_id):
                    pairs.append((adult, acct))
            pairs.sort(key=lambda p: (p[0].age < PENALTY_FREE_AGE,))
            for adult, acct in pairs:
                exempt = (tier == "college" and acct.kind == AccountKind.TRAD_IRA)
                raised += pull(acct, needed - raised, owner_age=adult.age,
                               penalty_exempt=exempt)
                if raised >= needed - 1e-9:
                    break
        elif tranche == "hsa":
            acct = state.account("hsa")
            if acct:
                raised += pull(acct, needed - raised)
        elif tranche == "roth_earnings":
            for adult in state.adults:
                acct = state.account(f"roth_ira_{adult.person_id}")
                if acct:
                    raised += pull(acct, needed - raised, owner_age=adult.age)
                if raised >= needed - 1e-9:
                    break
        elif tranche == "529":
            for child in state.children:
                acct = state.account(f"529_{child.person_id}")
                if acct:
                    raised += pull(acct, needed - raised)
                if raised >= needed - 1e-9:
                    break
    return raised


def _fund(state: HouseholdState, amount: float, T: TaxYearInput,
          tier: str, log: YearLog) -> float:
    """Pay ``amount`` from cash, cascading into forced withdrawals if short.

    Returns the amount actually funded (≤ amount)."""
    if amount <= 0:
        return 0.0
    paid = _take_cash(state, amount)
    if paid < amount - 1e-9:
        raised = _raise_cash(state, amount - paid, T, tier)
        log.forced_liquidation += raised
        paid += _take_cash(state, amount - paid)
    return paid


def _settle_taxes(state: HouseholdState, T: TaxYearInput, params: TaxParams,
                  already_paid: float, cfg: SimConfig, log: YearLog,
                  ) -> tuple[float, float]:
    """Pay (total tax owed on T) − already_paid, forcing liquidation if cash
    is short. Liquidation itself adds taxable income, so iterate to a fixed
    point. Returns (paid_now, unpaid)."""
    paid_now = 0.0
    for _ in range(cfg.settle_iterations):
        due = compute_taxes(T, params).total_tax - already_paid - paid_now
        if due <= 0.01:
            return paid_now, 0.0
        available = _cash(state).balance
        if available + 1e-9 >= due:
            _take_cash(state, due)
            paid_now += due
            continue  # paying didn't add income; next loop confirms fixed point
        # cash short: raise the difference (adds income → loop again)
        # raise a hair extra so gain-on-the-sale tax doesn't leave a residue
        got = _raise_cash(state, (due - available) * 1.02 + 1.0, T, tier="tax")
        log.forced_liquidation += got
        if got <= 0.01:
            # resources exhausted: pay what's left in cash, record unpaid
            paid_now += _take_cash(state, due)
            due_final = compute_taxes(T, params).total_tax - already_paid - paid_now
            return paid_now, (due_final if due_final > 0.5 else 0.0)
    due_final = compute_taxes(T, params).total_tax - already_paid - paid_now
    if due_final > 0.01:
        paid_now += _take_cash(state, due_final)
        due_final = compute_taxes(T, params).total_tax - already_paid - paid_now
    return paid_now, (due_final if due_final > 0.5 else 0.0)


def _pay_mortgage(state: HouseholdState, funds_available: float,
                  ) -> tuple[float, float]:
    """Run 12 months of amortization with at most ``funds_available`` to
    spend. Unpaid months accrue interest onto the balance (negative
    amortization stands in for arrears/penalties). Returns (paid, due)."""
    m = state.mortgage
    if m is None or not m.active:
        return 0.0, 0.0
    paid = 0.0
    due = 0.0
    r = m.annual_rate / 12.0
    for _ in range(12):
        if not m.active:
            break
        interest = m.balance * r
        payment = min(m.monthly_payment, m.balance + interest)
        due += payment
        if funds_available - paid >= payment - 1e-9:
            m.balance = max(0.0, m.balance + interest - payment)
            paid += payment
        else:
            m.balance += interest  # missed month
    return paid, due


def _mortgage_due_dry_run(state: HouseholdState) -> float:
    m = state.mortgage
    if m is None or not m.active:
        return 0.0
    bal = m.balance
    r = m.annual_rate / 12.0
    due = 0.0
    for _ in range(12):
        if bal <= 0.005:
            break
        interest = bal * r
        payment = min(m.monthly_payment, bal + interest)
        due += payment
        bal = max(0.0, bal + interest - payment)
    return due


# ---------------------------------------------------------------------------
# MAGI estimate (for the Roth phase-out check and the agent's observation)
# ---------------------------------------------------------------------------


def estimate_magi(state: HouseholdState, decision: Decision,
                  cfg: SimConfig) -> float:
    """Deterministic provisional MAGI for this year, computed before returns
    are known (they may not be used — no forward-looking information).

    Approximations, documented: investment income is estimated from current
    taxable balances at the configured yields; realized gains from the
    *voluntary* taxable withdrawal at the current basis ratio; Social
    Security at the 85% maximum-inclusion bound.
    """
    wages = sum(a.gross_salary for a in state.adults if a.working)
    pretax = min(decision.contributions.k401_a1 + decision.contributions.k401_a2
                 + decision.contributions.hsa, wages)
    taxable = state.account("taxable")
    invest_income = 0.0
    gains = 0.0
    if taxable and taxable.balance > 0:
        eq = taxable.balance * taxable.alloc.equity_fraction()
        bd = taxable.balance * taxable.alloc.bonds
        invest_income = eq * cfg.dividend_yield + bd * cfg.bond_income_yield
        gains = decision.withdrawals.taxable * (1.0 - taxable.basis_ratio())
    ss = sum(a.ss_annual_benefit for a in state.adults if a.ss_claimed)
    trad_wd = decision.withdrawals.trad_401k + decision.withdrawals.trad_ira
    return (wages - pretax + trad_wd + decision.roth_conversion_amount
            + invest_income + gains + 0.85 * ss)


# ---------------------------------------------------------------------------
# The annual step
# ---------------------------------------------------------------------------


def simulate_year(state: HouseholdState, decision: Decision,
                  yr: YearReturns, cfg: SimConfig,
                  prevalidated: bool = False,
                  ) -> tuple[HouseholdState, YearLog]:
    """Advance the household one year. Returns (new_state, log).

    ``decision`` may be raw agent output; it is validated/clipped here unless
    ``prevalidated`` (the runner validates first so the agent's violation
    count is attributed before any engine mechanics run — results identical).
    """
    st = state.clone()
    log = YearLog(year_index=st.year_index, calendar_year=st.calendar_year,
                  returns=yr.to_dict(), inflation_index_boy=st.inflation_index)

    params = TaxParams.for_year(st.filing_status, st.inflation_index,
                                enable_ctc=cfg.enable_ctc,
                                enable_niit=cfg.enable_niit)
    limits = ContributionLimits.for_year(st.inflation_index, st.filing_status)

    # ---- validation (rule feasibility) -----------------------------------
    if prevalidated:
        d, violations = decision, []
    else:
        d, violations = validate_decision(decision, st, limits,
                                          estimate_magi(state, decision, cfg))

    events = life.LifeEventLog()

    # ---- 1. life events at year start -------------------------------------
    retire_ids = {pid for pid, attr in (("a1", "adult1"), ("a2", "adult2"))
                  if getattr(d.retire_now, attr)}
    life.apply_retirement_transitions(st, retire_ids, events)
    claim_ids = {pid for pid, attr in (("a1", "adult1"), ("a2", "adult2"))
                 if getattr(d.claim_social_security, attr)}
    life.apply_ss_claims(st, claim_ids, events)
    life.college_transitions(st.children, events)
    log.ages = {a.person_id: a.age for a in st.adults}
    log.ages.update({c.person_id: c.age for c in st.children})
    log.all_retired = all(a.retired for a in st.adults)

    # RMD requirements come from start-of-year balances (≈ prior Dec 31),
    # measured BEFORE this year's flows.
    rmds = household_rmd_requirements(st)

    T = TaxYearInput(capital_loss_carryforward_in=st.capital_loss_carryforward,
                     num_ctc_children=sum(1 for c in st.children if c.age < 16))

    # ---- 2. income ----------------------------------------------------------
    wages = []
    for adult in st.adults:
        w = adult.gross_salary if adult.working else 0.0
        wages.append(w)
        log.salaries[adult.person_id] = w
    T.wages_by_adult = tuple(wages)
    _cash(st).balance += sum(wages)

    ss_total = sum(a.ss_annual_benefit for a in st.adults if a.ss_claimed)
    T.ss_benefits_gross = ss_total
    _cash(st).balance += ss_total
    log.ss_benefits = ss_total
    log.life_events = events.to_dict()

    # ---- 3a. payroll deferrals + employer match ------------------------------
    c = d.contributions
    applied_contribs: dict[str, float] = {}
    match_total = 0.0
    for adult, amt_attr in ((st.adult("a1"), "k401_a1"), (st.adult("a2"), "k401_a2")):
        if adult is None:
            continue
        contrib = getattr(c, amt_attr)
        # retirement may have just happened at start of year → no wages now
        if not adult.working:
            contrib = 0.0
        acct = st.account(f"trad_401k_{adult.person_id}")
        if acct is None or contrib <= 0:
            continue
        contrib = min(contrib, _cash(st).balance)  # wages already landed in cash
        _take_cash(st, contrib)
        match = employer_match(contrib, adult.gross_salary,
                               adult.match_rate, adult.match_cap_fraction)
        acct.balance += contrib + match
        T.pretax_401k += contrib
        match_total += match
        applied_contribs[f"401k_{adult.person_id}"] = contrib
    log.employer_match_total = match_total

    hsa_acct = st.account("hsa")
    if hsa_acct is not None and c.hsa > 0:
        amt = min(c.hsa, _cash(st).balance)
        if amt < c.hsa - 0.01:
            violations.append(Violation("insufficient_cash", "contributions.hsa",
                                        "HSA contribution reduced to available cash",
                                        proposed=c.hsa, clipped_to=amt))
        _take_cash(st, amt)
        hsa_acct.balance += amt
        T.hsa_contributions += amt
        applied_contribs["hsa"] = amt

    # ---- 3b. voluntary withdrawals -------------------------------------------
    w = d.withdrawals

    def _split_pro_rata(account_ids: list[str], total: float) -> dict[str, float]:
        balances = {aid: st.accounts[aid].balance for aid in account_ids
                    if aid in st.accounts and st.accounts[aid].balance > 0}
        pool = sum(balances.values())
        if pool <= 0 or total <= 0:
            return {}
        take = min(total, pool)
        return {aid: take * b / pool for aid, b in balances.items()}

    vol_wd: dict[str, float] = {}
    trad_taken_by_adult: dict[str, float] = {}

    if w.taxable > 0 and (acct := st.account("taxable")):
        eff = withdraw_from_account(acct, w.taxable, 99)
        T.realized_lt_gains += eff.realized_lt_gain
        _cash(st).balance += eff.gross
        vol_wd["taxable"] = eff.gross

    for field_name, ids in (("trad_401k", ["trad_401k_a1", "trad_401k_a2"]),
                            ("trad_ira", ["trad_ira_a1", "trad_ira_a2"]),
                            ("roth", ["roth_ira_a1", "roth_ira_a2"])):
        total = getattr(w, field_name)
        for aid, amt in _split_pro_rata(ids, total).items():
            acct = st.accounts[aid]
            age = _adult_age(st, acct.owner)
            eff = withdraw_from_account(acct, amt, age)
            if acct.kind in (AccountKind.TRAD_401K, AccountKind.TRAD_IRA):
                T.trad_taxable_withdrawals += eff.ordinary_income
                if acct.owner:
                    trad_taken_by_adult[acct.owner] = (
                        trad_taken_by_adult.get(acct.owner, 0.0) + eff.gross)
            elif acct.kind == AccountKind.ROTH_IRA and eff.ordinary_income > 0:
                T.other_ordinary_income += eff.ordinary_income
            T.penalty_base += eff.penalty_base
            _cash(st).balance += eff.gross
            vol_wd[field_name] = vol_wd.get(field_name, 0.0) + eff.gross

    if w.hsa > 0 and (acct := st.account("hsa")):
        eff = withdraw_from_account(acct, w.hsa, 99)  # assumed qualified
        _cash(st).balance += eff.gross
        vol_wd["hsa"] = eff.gross

    # ---- 3c. RMD enforcement (before conversions; conversions never satisfy
    #          an RMD) -----------------------------------------------------------
    rmd_forced_total = 0.0
    for adult in st.adults:
        required = rmds.get(adult.person_id, 0.0)
        if required <= 0:
            continue
        shortfall = required - trad_taken_by_adult.get(adult.person_id, 0.0)
        if shortfall > 0.01:
            for acct in st.adult_trad_accounts(adult.person_id):
                if shortfall <= 0.01:
                    break
                eff = withdraw_from_account(acct, shortfall, adult.age)
                T.trad_taxable_withdrawals += eff.ordinary_income
                _cash(st).balance += eff.gross
                rmd_forced_total += eff.gross
                shortfall -= eff.gross
    log.rmd_forced = rmd_forced_total

    # ---- 3d. Roth conversions ---------------------------------------------------
    conversion_left = d.roth_conversion_amount
    converted_total = 0.0
    if conversion_left > 0:
        source_ids = ["trad_ira_a1", "trad_ira_a2", "trad_401k_a1", "trad_401k_a2"]
        for aid in source_ids:
            if conversion_left <= 0.01:
                break
            src = st.account(aid)
            if src is None or src.balance <= 0 or src.owner is None:
                continue
            roth = st.account(f"roth_ira_{src.owner}")
            if roth is None:
                continue
            eff = roth_convert(src, roth, conversion_left)
            T.roth_conversion_taxable += eff.taxable_ordinary
            converted_total += eff.converted
            conversion_left -= eff.converted
    log.roth_converted = converted_total

    # ---- 3e. after-tax contributions from cash ----------------------------------
    def _after_tax(field_label: str, requested: float) -> float:
        if requested <= 0:
            return 0.0
        amt = min(requested, _cash(st).balance)
        if amt < requested - 0.01:
            violations.append(Violation("insufficient_cash", field_label,
                                        "contribution reduced to available cash",
                                        proposed=requested, clipped_to=amt))
        return amt

    for pid, amt_attr, flag_attr in (("a1", "roth_ira_a1", "backdoor_roth_a1"),
                                     ("a2", "roth_ira_a2", "backdoor_roth_a2")):
        roth = st.account(f"roth_ira_{pid}")
        if roth is None:
            continue
        amt = _after_tax(f"contributions.roth_ira_{pid}", getattr(c, amt_attr))
        if amt <= 0:
            continue
        _take_cash(st, amt)
        if getattr(c, flag_attr):
            eff = backdoor_roth_contribution(st.account(f"trad_ira_{pid}"), roth, amt)
            T.roth_conversion_taxable += eff.taxable_ordinary
            applied_contribs[f"backdoor_roth_{pid}"] = amt
        else:
            contribute_to_account(roth, amt)
            applied_contribs[f"roth_ira_{pid}"] = amt

    for cid, amt_attr in (("child1", "c529_child1"), ("child2", "c529_child2")):
        acct = st.account(f"529_{cid}")
        if acct is None:
            continue
        amt = _after_tax(f"contributions.529_{cid}", getattr(c, amt_attr))
        if amt > 0:
            _take_cash(st, amt)
            contribute_to_account(acct, amt)
            applied_contribs[f"529_{cid}"] = amt

    if (acct := st.account("taxable")) is not None:
        amt = _after_tax("contributions.taxable", c.taxable)
        if amt > 0:
            _take_cash(st, amt)
            contribute_to_account(acct, amt)
            applied_contribs["taxable"] = amt

    if st.mortgage is not None and st.mortgage.active and c.extra_mortgage_principal > 0:
        amt = _after_tax("contributions.extra_mortgage_principal",
                         min(c.extra_mortgage_principal, st.mortgage.balance))
        if amt > 0:
            _take_cash(st, amt)
            st.mortgage.balance -= amt
            applied_contribs["extra_mortgage_principal"] = amt

    log.contributions_applied = applied_contribs
    log.withdrawals_voluntary = vol_wd

    # ---- 3f. tax-loss harvest (realizes the unrealized loss, resets basis) -------
    if d.tax_loss_harvest and (acct := st.account("taxable")) is not None:
        loss = acct.basis - acct.balance
        if loss > 0.01:
            T.realized_lt_gains -= loss
            acct.basis = acct.balance

    # ---- 3g. rebalance to target allocations --------------------------------------
    for aid, weights in d.allocations.items():
        acct = st.account(aid)
        if acct is not None and acct.kind != AccountKind.CASH:
            acct.alloc = Allocation.from_dict(weights)
    log.allocations_target = {aid: a.alloc.as_dict() for aid, a in st.accounts.items()}

    # ---- 4. market returns ----------------------------------------------------
    asset_returns = yr.asset_dict()
    for acct in st.accounts.values():
        if acct.balance <= 0:
            continue
        pre = acct.balance
        if acct.kind == AccountKind.CASH:
            r = asset_returns["cash"]
            interest = pre * max(0.0, r)
            T.interest_ordinary += interest
            acct.balance = pre * (1.0 + r)
        else:
            if acct.kind == AccountKind.TAXABLE:
                eq = pre * acct.alloc.equity_fraction()
                bd = pre * acct.alloc.bonds
                csh = pre * acct.alloc.cash
                div = eq * cfg.dividend_yield
                bond_int = bd * cfg.bond_income_yield
                cash_int = csh * max(0.0, asset_returns["cash"])
                T.qualified_dividends += div
                T.interest_ordinary += bond_int + cash_int
                # distributions are reinvested after being taxed → basis grows
                acct.basis += div + bond_int + cash_int
            acct.balance = pre * (1.0 + acct.alloc.weighted_return(asset_returns))

    # ---- 5. tax settlement #1 ----------------------------------------------------
    paid_1, unpaid_1 = _settle_taxes(st, T, params, 0.0, cfg, log)

    # ---- 6. expenses in priority order ---------------------------------------------
    exp: dict[str, float] = {}

    essential_due = st.essential_expenses
    essential_paid = _fund(st, essential_due, T, "essential", log)
    if essential_paid < essential_due - 0.5:
        log.shortfalls.append({"tier": "essential",
                               "amount": essential_due - essential_paid})
    exp["essential_due"] = essential_due
    exp["essential_paid"] = essential_paid

    mortgage_due = _mortgage_due_dry_run(st)
    mortgage_paid = 0.0
    if mortgage_due > 0:
        funded = _fund(st, mortgage_due, T, "mortgage", log)
        mortgage_paid, _due = _pay_mortgage(st, funded)
        leftover = funded - mortgage_paid
        if leftover > 0.01:
            _cash(st).balance += leftover
        if mortgage_paid < mortgage_due - 0.5:
            log.shortfalls.append({"tier": "mortgage",
                                   "amount": mortgage_due - mortgage_paid})
    exp["mortgage_due"] = mortgage_due
    exp["mortgage_paid"] = mortgage_paid

    college_bills = life.college_costs_due(st)
    college_due = sum(college_bills.values())
    college_paid = 0.0
    college_from_529 = 0.0
    college_by_child: dict[str, dict[str, float]] = {}
    for cid, bill in college_bills.items():
        remaining = bill
        paid_child = 0.0
        acct = st.account(f"529_{cid}")
        if acct is not None and acct.balance > 0:
            eff = withdraw_from_account(acct, remaining, 99, qualified_529=True)
            college_from_529 += eff.gross
            paid_child += eff.gross
            remaining -= eff.gross
        if remaining > 0.01:
            paid = _fund(st, remaining, T, "college", log)
            paid_child += paid
            if paid < remaining - 0.5:
                log.shortfalls.append({"tier": "college", "child": cid,
                                       "amount": remaining - paid})
        college_paid += paid_child
        college_by_child[cid] = {"due": bill, "paid": paid_child}
    exp["college_due"] = college_due
    exp["college_paid"] = college_paid
    exp["college_from_529"] = college_from_529
    exp["college_by_child"] = college_by_child

    disc_target = d.annual_spending_discretionary
    disc_paid = _fund(st, disc_target, T, "discretionary", log)
    exp["discretionary_target"] = disc_target
    exp["discretionary_paid"] = disc_paid
    log.expenses = exp

    # ---- 7. tax true-up -------------------------------------------------------------
    paid_2, unpaid_2 = _settle_taxes(st, T, params, paid_1, cfg, log)
    final_tax = compute_taxes(T, params)
    log.unpaid_tax = unpaid_2
    if unpaid_2 > 0.5:
        log.shortfalls.append({"tier": "tax", "amount": unpaid_2})
    st.capital_loss_carryforward = final_tax.capital_loss_carryforward_out
    log.tax = {
        "agi": final_tax.agi,
        "taxable_income": final_tax.taxable_income,
        "ss_taxable": final_tax.ss_taxable,
        "ordinary_tax": final_tax.ordinary_tax,
        "ltcg_tax": final_tax.ltcg_tax,
        "ctc": final_tax.ctc,
        "niit": final_tax.niit,
        "income_tax": final_tax.income_tax,
        "payroll_tax": final_tax.payroll_tax,
        "penalties": final_tax.penalties,
        "total_tax": final_tax.total_tax,
        "tax_paid": paid_1 + paid_2,
        "marginal_ordinary_rate": final_tax.marginal_ordinary_rate,
        "loss_carryforward_out": final_tax.capital_loss_carryforward_out,
    }

    # ---- 8. advance -------------------------------------------------------------------
    infl = yr.inflation
    log.decision = d.to_json_dict()
    log.rationale = d.rationale
    log.violations = [x.to_dict() for x in violations]
    log.consumption_nominal = essential_paid + disc_paid
    # deflate by the start-of-year index — the basis on which this year's
    # expenses were quoted
    log.consumption_real = log.consumption_nominal / st.inflation_index

    for adult in st.adults:
        adult.age += 1
        if adult.working:
            adult.gross_salary *= (1.0 + adult.salary_growth)
        adult.ss_pia_annual *= (1.0 + infl)  # COLA (pre-claim wage indexing ≈ CPI)
        if adult.ss_claimed:
            adult.ss_annual_benefit *= (1.0 + infl)
    for child in st.children:
        child.age += 1
        child.college_annual_cost *= (1.0 + infl + cfg.college_inflation_premium)

    st.essential_expenses *= (1.0 + infl)
    st.discretionary_floor *= (1.0 + infl)
    st.discretionary_ceiling *= (1.0 + infl)
    st.inflation_index *= (1.0 + infl)
    st.year_index += 1
    st.prior_decision = d.to_json_dict()
    st.prior_rationale = d.rationale
    st.returns_history.append(yr.to_dict())
    if len(st.returns_history) > 10:
        st.returns_history = st.returns_history[-10:]

    log.eoy_balances = {aid: round(a.balance, 2) for aid, a in st.accounts.items()}
    log.mortgage_balance = st.mortgage.balance if st.mortgage else 0.0
    log.net_worth_nominal = st.net_worth()
    log.net_worth_real = st.net_worth() / st.inflation_index
    log.inflation_index_eoy = st.inflation_index

    return st, log


# ---------------------------------------------------------------------------
# Observation building (what an agent is allowed to see)
# ---------------------------------------------------------------------------


def _rules_text(state: HouseholdState, limits: ContributionLimits) -> str:
    ages = {a.person_id: a.age for a in state.adults}
    lines = [
        "You decide once per year. Decision fields (all dollar amounts are "
        "this year's nominal dollars):",
        "- annual_spending_discretionary: total discretionary spending, within "
        f"[{state.discretionary_floor:,.0f}, {state.discretionary_ceiling:,.0f}].",
        "- contributions: 401k_a1/401k_a2 (employee deferral, requires wages; "
        "employer match per the household's match rules is added automatically), "
        "roth_ira_a1/roth_ira_a2 (+backdoor_roth flags to bypass the MAGI "
        "phase-out via backdoor; pro-rata taxation applies if pre-tax IRA "
        "balances exist), hsa, 529_child1/529_child2, taxable, "
        "extra_mortgage_principal.",
        "- roth_conversion_amount: converts pre-tax dollars (IRAs first, then "
        "401(k)s) to Roth; taxed as ordinary income, no early penalty.",
        "- withdrawals: taxable / trad_401k / trad_ira / roth / hsa. Trad "
        "withdrawals are ordinary income; 10% penalty when the owner is under "
        "59.5. Roth withdrawals return contribution basis first (tax-free).",
        "- allocations: per-account target weights over stocks_us, stocks_intl, "
        "bonds, cash (sum to 1; omitted accounts keep prior targets; the cash "
        "account is always cash).",
        "- retire_now / claim_social_security: one-shot flags per adult. SS is "
        "claimable from 62 (70% of PIA) to 70 (124%); FRA is 67. RMDs from 73 "
        "are auto-enforced.",
        "- tax_loss_harvest: realize taxable-account losses (if any) this year.",
        "Everything infeasible is clipped to the nearest feasible value and "
        "recorded as a violation — violations count against you.",
        "Expense mechanics: essential expenses, mortgage, and college bills are "
        "paid automatically (college draws the child's 529 first). If cash runs "
        "short, the engine force-liquidates in the order taxable → Roth basis → "
        "pre-tax (with penalties if young) → HSA → Roth earnings → 529, and "
        "records shortfall events if even that fails.",
        "Taxes: federal MFJ/HoH with standard deduction, LTCG stacking, payroll "
        "taxes, SS provisional-income taxation, CTC, NIIT. Taxes are paid from "
        "cash in the same year.",
    ]
    if ages:
        lines.append(f"Current ages: {ages}.")
    return "\n".join(lines)


def _goals_text(state: HouseholdState) -> str:
    g = state.goals
    lines = [
        "  1. NEVER run out of money: essential spending (and mortgage) funded "
        "every year until both adults reach 95.",
    ]
    if state.children:
        lines.append(f"  2. Fund at least {g.college_fund_fraction:.0%} of each "
                     "child's college cost (4 years each unless noted).")
    lines.append(f"  3. Retirement lifestyle: total spending of "
                 f"${g.retirement_spending_real:,.0f}/yr in today's dollars "
                 "throughout retirement.")
    if g.bequest_target_real:
        lines.append(f"  4. Leave a bequest of at least "
                     f"${g.bequest_target_real:,.0f} (today's dollars).")
    return "\n".join(lines)


def _upcoming_obligations(state: HouseholdState, cfg: SimConfig) -> list[str]:
    out: list[str] = []
    for child in state.children:
        yrs = child.college_start_age - child.age
        cost_today = state.real(child.college_annual_cost)
        if 0 < yrs:
            projected = cost_today * (1 + cfg.college_inflation_premium) ** yrs
            out.append(f"{child.person_id} starts college in {yrs} years "
                       f"({child.college_years} years long); projected cost "
                       f"~${projected:,.0f}/yr in today's dollars "
                       f"(college inflation runs CPI + "
                       f"{cfg.college_inflation_premium:.1%}).")
        elif child.in_college():
            remaining = child.college_end_age - child.age
            out.append(f"{child.person_id} is IN college: "
                       f"${child.college_annual_cost:,.0f} due this year, "
                       f"{remaining} year(s) remaining including this one.")
    for adult in state.adults:
        if not adult.retired:
            yrs = adult.retirement_age - adult.age
            if yrs > 0:
                out.append(f"{adult.person_id} plans to retire in {yrs} years "
                           f"(age {adult.retirement_age}); you may retire them "
                           "earlier/later via retire_now or by doing nothing.")
        if not adult.ss_claimed:
            if adult.age < 62:
                out.append(f"{adult.person_id} can claim Social Security from age "
                           f"62 (in {62 - adult.age} years).")
    if state.mortgage is not None and state.mortgage.active:
        m = state.mortgage
        annual = m.monthly_payment * 12
        out.append(f"Mortgage: ${m.balance:,.0f} at {m.annual_rate:.2%}, "
                   f"~${annual:,.0f}/yr P&I until paid off.")
    return out


def build_observation(state: HouseholdState, cfg: SimConfig,
                      scenario_name: str = "") -> Observation:
    """Assemble the agent-facing observation for the current year.

    Strictly backward-looking. The MAGI estimate uses a no-action decision
    (the agent's own moves will shift actual MAGI; the validator re-estimates
    with the real decision when clipping the Roth phase-out)."""
    limits = ContributionLimits.for_year(state.inflation_index, state.filing_status)
    ages = [a.age for a in state.adults]
    limits_dict = {
        "401k_employee_per_adult (age<50)": limits.limit_401k_employee,
        "401k_catchup_50plus": limits.limit_401k_catchup,
        "ira_per_adult (age<50)": limits.limit_ira,
        "ira_catchup_50plus": limits.limit_ira_catchup,
        "hsa_family": limits.max_hsa(ages) if state.hsa_enabled else 0.0,
        "529_per_child": limits.limit_529_per_child,
        "roth_magi_phaseout_start": limits.roth_magi_phaseout_lo,
        "roth_magi_phaseout_end": limits.roth_magi_phaseout_hi,
    }
    return Observation(
        year_index=state.year_index,
        calendar_year=state.calendar_year,
        state=state.to_dict(),
        limits=limits_dict,
        rules_text=_rules_text(state, limits),
        upcoming=_upcoming_obligations(state, cfg),
        goals_text=_goals_text(state),
        recent_returns=state.returns_history[-5:],
        prior_decision=state.prior_decision,
        prior_rationale=state.prior_rationale,
        estimated_magi=estimate_magi(state, Decision(), cfg),
        rmd_due=household_rmd_requirements(state),
        scenario_name=scenario_name,
    )
