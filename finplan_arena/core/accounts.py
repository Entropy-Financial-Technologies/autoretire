"""Account rules: contribution limits, RMDs, early-withdrawal penalties,
Social Security claiming factors, and withdrawal tax characterization.

2025 baseline limits (Rev. Proc. 2024-40 / IRS Notice 2024-80), indexed by
realized simulated CPI. We index smoothly (no $500-increment rounding) and
also index the IRA catch-up (fixed in law) — both documented simplifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .state import Account, AccountKind, HouseholdState, Person

# ---------------------------------------------------------------------------
# Ages (annual-timestep conventions, documented in README)
# ---------------------------------------------------------------------------

#: Withdrawals in a year are penalty-free if start-of-year age >= 59
#: (the person passes 59.5 during that year).
PENALTY_FREE_AGE = 59

#: RMDs start in the year the person's start-of-year age reaches 73
#: (SECURE 2.0; the age-75 step-up in 2033 is not modeled).
RMD_START_AGE = 73

#: HSA contributions stop when the *older* adult reaches 65 (Medicare).
HSA_MAX_AGE = 65

SS_EARLIEST_CLAIM_AGE = 62
SS_LATEST_CLAIM_AGE = 70
SS_FULL_RETIREMENT_AGE = 67


# ---------------------------------------------------------------------------
# Contribution limits (2025 baseline, CPI-indexed in-sim)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContributionLimits:
    """Per-year dollar limits, already inflation-indexed."""

    limit_401k_employee: float
    limit_401k_catchup: float       # age 50+ (the 60-63 "super" catch-up is not modeled)
    limit_ira: float                # Roth + Trad combined, per person
    limit_ira_catchup: float        # age 50+
    limit_hsa_family: float
    limit_hsa_catchup: float        # age 55+, per covered adult
    limit_529_per_child: float      # 2x annual gift exclusion (MFJ); proxy cap
    roth_magi_phaseout_lo: float
    roth_magi_phaseout_hi: float

    @classmethod
    def for_year(cls, index_factor: float, filing_status: str = "mfj") -> "ContributionLimits":
        k = index_factor
        # Roth phase-out: MFJ $236k-$246k; single/HoH $150k-$165k (2025).
        if filing_status == "mfj":
            lo, hi = 236_000.0, 246_000.0
            gift_donors = 2
        else:
            lo, hi = 150_000.0, 165_000.0
            gift_donors = 1
        return cls(
            limit_401k_employee=23_500.0 * k,
            limit_401k_catchup=7_500.0 * k,
            limit_ira=7_000.0 * k,
            limit_ira_catchup=1_000.0 * k,
            limit_hsa_family=8_550.0 * k,
            limit_hsa_catchup=1_000.0 * k,
            limit_529_per_child=19_000.0 * gift_donors * k,
            roth_magi_phaseout_lo=lo * k,   # Roth phase-out IS indexed in law
            roth_magi_phaseout_hi=hi * k,
        )

    def max_401k_employee(self, age: int) -> float:
        return self.limit_401k_employee + (self.limit_401k_catchup if age >= 50 else 0.0)

    def max_ira(self, age: int) -> float:
        return self.limit_ira + (self.limit_ira_catchup if age >= 50 else 0.0)

    def max_hsa(self, adult_ages: list[int]) -> float:
        cap = self.limit_hsa_family
        for a in adult_ages:
            if a >= 55:
                cap += self.limit_hsa_catchup
        return cap

    def roth_ira_allowed_fraction(self, magi: float) -> float:
        """Fraction of the IRA limit contributable directly at this MAGI
        (1.0 below the phase-out, 0.0 above it, linear in between)."""
        if magi <= self.roth_magi_phaseout_lo:
            return 1.0
        if magi >= self.roth_magi_phaseout_hi:
            return 0.0
        span = self.roth_magi_phaseout_hi - self.roth_magi_phaseout_lo
        return (self.roth_magi_phaseout_hi - magi) / span


def employer_match(employee_contribution: float, salary: float,
                   match_rate: float, match_cap_fraction: float) -> float:
    """Employer match: ``match_rate`` of contributions up to
    ``match_cap_fraction`` of salary (e.g. 50% of the first 6%)."""
    if salary <= 0:
        return 0.0
    return match_rate * min(employee_contribution, match_cap_fraction * salary)


# ---------------------------------------------------------------------------
# RMDs — IRS Uniform Lifetime Table (2022+)
# ---------------------------------------------------------------------------

_UNIFORM_LIFETIME = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0,
    79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0,
    86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8,
    93: 10.1, 94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8,
    100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6,
}


def rmd_factor(age: int) -> float:
    if age < RMD_START_AGE:
        return 0.0
    return _UNIFORM_LIFETIME.get(min(age, 105), 4.6)


def required_minimum_distribution(age: int, trad_balance_start_of_year: float) -> float:
    """RMD owed this year, based on the prior Dec-31 balance (== start-of-year
    balance in this engine, since flows and returns happen mid-step).

    Simplification: each adult's 401(k) and trad IRA are aggregated into one
    RMD (real rules require separate 401(k) computation; same total here).
    The first-year April-1 deferral option is not modeled.
    """
    f = rmd_factor(age)
    if f <= 0 or trad_balance_start_of_year <= 0:
        return 0.0
    return trad_balance_start_of_year / f


# ---------------------------------------------------------------------------
# Social Security claiming
# ---------------------------------------------------------------------------


def ss_claiming_factor(claim_age: int, fra: int = SS_FULL_RETIREMENT_AGE) -> float:
    """Benefit as a fraction of PIA for claiming at ``claim_age``.

    Piecewise-linear approximation anchored at the statutory points for
    FRA 67: 70% at 62, 100% at 67, 124% at 70 (spec-sanctioned approximation
    of the 5/9%-per-month early reduction and 8%/yr delayed credits).
    """
    a = max(SS_EARLIEST_CLAIM_AGE, min(SS_LATEST_CLAIM_AGE, claim_age))
    if a < fra:
        return 1.0 - 0.06 * (fra - a)
    return 1.0 + 0.08 * (a - fra)


# ---------------------------------------------------------------------------
# Withdrawal characterization
# ---------------------------------------------------------------------------


@dataclass
class WithdrawalTaxEffect:
    """Tax characterization of a single-account withdrawal (gross amount)."""

    gross: float = 0.0
    ordinary_income: float = 0.0     # taxable-as-ordinary portion
    realized_lt_gain: float = 0.0    # taxable account only
    penalty_base: float = 0.0        # subject to the 10% penalty
    tax_free: float = 0.0


def withdraw_from_account(acct: Account, amount: float, owner_age: int,
                          *, qualified_529: bool = False,
                          penalty_exempt: bool = False) -> WithdrawalTaxEffect:
    """Remove up to ``amount`` from ``acct`` and characterize it for tax.

    Mutates the account (balance and basis). Returns the effect with
    ``gross`` = the amount actually withdrawn (capped by the balance).

    Rules implemented:
    * TAXABLE — pro-rata blended basis; the gain slice is LTG (all holdings
      are assumed long-term; see README).
    * TRAD_401K — fully ordinary; 10% penalty if under 59.5 (age proxy) and
      not ``penalty_exempt``. (Rule-of-55 not modeled.)
    * TRAD_IRA — ordinary except the pro-rata after-tax basis slice
      (backdoor leftovers); same penalty rule. ``penalty_exempt`` covers the
      IRA higher-education exception used by the college funding cascade.
    * ROTH_IRA — basis (contributions + conversions) comes out first, tax-
      and penalty-free; earnings beyond basis are ordinary + penalty if
      under 59.5, tax-free otherwise. (5-year clocks not modeled.)
    * HSA — assumed qualified-medical, hence tax-free at any age (documented
      generous simplification).
    * COLLEGE_529 — tax-free when ``qualified_529``; otherwise the pro-rata
      earnings slice is ordinary income + 10% penalty (any owner age).
    * CASH — never taxable at withdrawal (interest is taxed as earned).
    """
    eff = WithdrawalTaxEffect()
    take = min(max(0.0, amount), acct.balance)
    if take <= 0:
        return eff
    eff.gross = take
    penalized = owner_age < PENALTY_FREE_AGE and not penalty_exempt

    if acct.kind == AccountKind.TAXABLE:
        ratio = acct.basis_ratio()
        eff.realized_lt_gain = take * (1.0 - ratio)
        acct.basis = max(0.0, acct.basis - take * ratio)
    elif acct.kind == AccountKind.TRAD_401K:
        eff.ordinary_income = take
        if penalized:
            eff.penalty_base = take
    elif acct.kind == AccountKind.TRAD_IRA:
        ratio = acct.basis_ratio()  # after-tax basis fraction
        basis_part = take * ratio
        taxable_part = take - basis_part
        acct.basis = max(0.0, acct.basis - basis_part)
        eff.ordinary_income = taxable_part
        eff.tax_free = basis_part
        if penalized:
            eff.penalty_base = taxable_part
    elif acct.kind == AccountKind.ROTH_IRA:
        basis_part = min(take, acct.basis)
        earnings_part = take - basis_part
        acct.basis -= basis_part
        eff.tax_free = basis_part
        if earnings_part > 0 and penalized:
            eff.ordinary_income = earnings_part
            eff.penalty_base = earnings_part
        else:
            eff.tax_free += earnings_part
    elif acct.kind == AccountKind.HSA:
        eff.tax_free = take
    elif acct.kind == AccountKind.COLLEGE_529:
        if qualified_529:
            ratio = acct.basis_ratio()
            acct.basis = max(0.0, acct.basis - take * ratio)
            eff.tax_free = take
        else:
            ratio = acct.basis_ratio()
            basis_part = take * ratio
            earnings = take - basis_part
            acct.basis = max(0.0, acct.basis - basis_part)
            eff.ordinary_income = earnings
            eff.penalty_base = earnings  # 529 10% penalty applies to earnings
            eff.tax_free = basis_part
    elif acct.kind == AccountKind.CASH:
        eff.tax_free = take
    else:  # pragma: no cover
        raise ValueError(f"Unknown account kind {acct.kind}")

    acct.balance -= take
    if acct.balance < 1e-9:
        acct.balance = 0.0
        if acct.kind != AccountKind.ROTH_IRA:
            acct.basis = 0.0
    return eff


def contribute_to_account(acct: Account, amount: float) -> None:
    """Add an after-tax (or payroll pre-tax) contribution to an account,
    maintaining basis semantics per kind."""
    if amount <= 0:
        return
    acct.balance += amount
    if acct.kind in (AccountKind.TAXABLE, AccountKind.ROTH_IRA,
                     AccountKind.COLLEGE_529):
        acct.basis += amount


@dataclass
class ConversionEffect:
    converted: float = 0.0
    taxable_ordinary: float = 0.0


def roth_convert(source: Account, roth: Account, amount: float) -> ConversionEffect:
    """Convert up to ``amount`` from a pre-tax account into a Roth IRA.

    Taxable as ordinary income except a trad IRA's pro-rata after-tax basis
    slice. No early-withdrawal penalty applies to conversions. The full
    converted amount becomes Roth basis (it is post-tax once converted;
    5-year recapture clocks are not modeled)."""
    eff = ConversionEffect()
    take = min(max(0.0, amount), source.balance)
    if take <= 0:
        return eff
    if source.kind == AccountKind.TRAD_IRA:
        ratio = source.basis_ratio()
        basis_part = take * ratio
        source.basis = max(0.0, source.basis - basis_part)
        eff.taxable_ordinary = take - basis_part
    elif source.kind == AccountKind.TRAD_401K:
        eff.taxable_ordinary = take
    else:
        raise ValueError(f"Cannot Roth-convert from {source.kind}")
    source.balance -= take
    if source.balance < 1e-9:
        source.balance = 0.0
        source.basis = 0.0
    roth.balance += take
    roth.basis += take
    eff.converted = take
    return eff


def backdoor_roth_contribution(trad_ira: Optional[Account], roth: Account,
                               amount: float) -> ConversionEffect:
    """Backdoor Roth: nondeductible trad-IRA contribution + immediate
    conversion, with the pro-rata rule.

    If the adult holds pre-tax trad-IRA dollars (balance P, after-tax basis B),
    a converted contribution C is taxable on the pro-rata pre-tax share:
    ``taxable = C * (P + C_pretax_excess...)`` — concretely, the conversion's
    nontaxable fraction is total_basis / total_balance measured after the
    contribution lands. The un-recovered basis stays behind in the trad IRA
    (exactly the real Form 8606 outcome, at annual granularity).
    """
    eff = ConversionEffect()
    if amount <= 0:
        return eff
    if trad_ira is None or trad_ira.balance <= 0:
        # clean backdoor: fully non-taxable
        roth.balance += amount
        roth.basis += amount
        eff.converted = amount
        return eff
    trad_ira.balance += amount
    trad_ira.basis += amount
    nontax_fraction = min(1.0, trad_ira.basis / trad_ira.balance)
    basis_recovered = amount * nontax_fraction
    eff.taxable_ordinary = amount - basis_recovered
    trad_ira.balance -= amount
    trad_ira.basis = max(0.0, trad_ira.basis - basis_recovered)
    roth.balance += amount
    roth.basis += amount
    eff.converted = amount
    return eff


def household_rmd_requirements(state: HouseholdState) -> dict[str, float]:
    """RMD owed per adult this year, from start-of-year trad balances."""
    out: dict[str, float] = {}
    for adult in state.adults:
        total = sum(a.balance for a in state.adult_trad_accounts(adult.person_id))
        rmd = required_minimum_distribution(adult.age, total)
        if rmd > 0:
            out[adult.person_id] = rmd
    return out
