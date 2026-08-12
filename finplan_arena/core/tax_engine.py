"""Federal tax engine (simplified but real).

Implements, for Married Filing Jointly and Head of Household:

* ordinary-income brackets (2025 baseline, inflation-indexed inside the sim)
* the standard deduction (itemizing is never modeled)
* long-term capital gains + qualified dividends stacked on ordinary income
  (0/15/20% brackets)
* payroll taxes (Social Security up to the wage base, Medicare, and the 0.9%
  Additional Medicare surtax — employee side only)
* Social Security benefit taxation via the two-threshold provisional-income
  worksheet (thresholds are *not* indexed, matching actual law)
* capital-loss netting with a $3,000/yr ordinary offset and indefinite
  carryforward
* the Child Tax Credit (nonrefundable simplification, $400k/$200k phase-out,
  thresholds unindexed per current law)
* NIIT (3.8% on net investment income over an unindexed MAGI threshold)
* 10% early-withdrawal penalties (computed from a penalty base supplied by
  the account layer)

Deliberately skipped (extension points, see README): state taxes, AMT,
itemized deductions, IRMAA, the refundable portion of the CTC, QBI, and the
0.9%-withholding true-up mechanics.

2025 baseline constants follow Rev. Proc. 2024-40 (pre-OBBBA figures).
Within the simulation they are indexed by *realized simulated* CPI, which is
the point: bracket creep interacts with the agent's Roth-conversion and
withdrawal choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

# ---------------------------------------------------------------------------
# Parameters (2025 baseline)
# ---------------------------------------------------------------------------

#: bracket tables: list of (top_of_bracket, rate); last top is infinity
_MFJ_ORDINARY_2025 = [
    (23_850.0, 0.10),
    (96_950.0, 0.12),
    (206_700.0, 0.22),
    (394_600.0, 0.24),
    (501_050.0, 0.32),
    (751_600.0, 0.35),
    (float("inf"), 0.37),
]

_HOH_ORDINARY_2025 = [
    (17_000.0, 0.10),
    (64_850.0, 0.12),
    (103_350.0, 0.22),
    (197_300.0, 0.24),
    (250_500.0, 0.32),
    (626_350.0, 0.35),
    (float("inf"), 0.37),
]

#: LTCG thresholds: (top of 0% bracket, top of 15% bracket)
_MFJ_LTCG_2025 = (96_700.0, 600_050.0)
_HOH_LTCG_2025 = (64_750.0, 566_700.0)

_STD_DEDUCTION_2025 = {"mfj": 30_000.0, "hoh": 22_500.0}

#: Social Security benefit taxation thresholds (fixed nominal — unindexed
#: by law since 1983/1993, which matters a lot over a 30-year horizon).
_SS_TAX_THRESHOLDS = {"mfj": (32_000.0, 44_000.0), "hoh": (25_000.0, 34_000.0)}

#: Additional Medicare / NIIT MAGI thresholds (unindexed by law).
_ADDL_MEDICARE_THRESHOLD = {"mfj": 250_000.0, "hoh": 200_000.0}
_NIIT_THRESHOLD = {"mfj": 250_000.0, "hoh": 200_000.0}

#: CTC phase-out start (unindexed by law).
_CTC_PHASEOUT_START = {"mfj": 400_000.0, "hoh": 200_000.0}


@dataclass(frozen=True)
class TaxParams:
    """Tax-law constants for one sim year (already inflation-indexed).

    ``index_factor`` is the cumulative realized-CPI factor applied to the
    2025 baseline for indexed quantities. Unindexed quantities (SS taxation
    thresholds, NIIT/Additional-Medicare/CTC thresholds, the $3k capital-loss
    ordinary offset) stay at their statutory nominal values — this is real
    law, not an oversight, and it creates genuine long-horizon tax drag.
    """

    filing_status: str = "mfj"
    index_factor: float = 1.0

    # indexed
    ordinary_brackets: list[tuple[float, float]] = field(default_factory=list)
    ltcg_0_top: float = 0.0
    ltcg_15_top: float = 0.0
    std_deduction: float = 0.0
    ss_wage_base: float = 0.0

    # unindexed (statutory nominal)
    ss_tax_t1: float = 0.0
    ss_tax_t2: float = 0.0
    addl_medicare_threshold: float = 0.0
    niit_threshold: float = 0.0
    ctc_per_child: float = 2_000.0
    ctc_phaseout_start: float = 0.0
    capital_loss_ordinary_offset: float = 3_000.0

    # rates
    ss_rate: float = 0.062
    medicare_rate: float = 0.0145
    addl_medicare_rate: float = 0.009
    niit_rate: float = 0.038
    ltcg_top_rate: float = 0.20
    early_withdrawal_penalty_rate: float = 0.10

    # feature toggles
    enable_ctc: bool = True
    enable_niit: bool = True

    @classmethod
    def for_year(cls, filing_status: str = "mfj", index_factor: float = 1.0,
                 enable_ctc: bool = True, enable_niit: bool = True) -> "TaxParams":
        fs = filing_status.lower()
        if fs not in ("mfj", "hoh"):
            raise ValueError(f"Unsupported filing status: {filing_status}")
        base = _MFJ_ORDINARY_2025 if fs == "mfj" else _HOH_ORDINARY_2025
        ltcg = _MFJ_LTCG_2025 if fs == "mfj" else _HOH_LTCG_2025
        k = index_factor
        return cls(
            filing_status=fs,
            index_factor=k,
            ordinary_brackets=[(top * k if top != float("inf") else top, r)
                               for top, r in base],
            ltcg_0_top=ltcg[0] * k,
            ltcg_15_top=ltcg[1] * k,
            std_deduction=_STD_DEDUCTION_2025[fs] * k,
            # The SS wage base is wage-indexed in law; we approximate with CPI.
            ss_wage_base=176_100.0 * k,
            ss_tax_t1=_SS_TAX_THRESHOLDS[fs][0],
            ss_tax_t2=_SS_TAX_THRESHOLDS[fs][1],
            addl_medicare_threshold=_ADDL_MEDICARE_THRESHOLD[fs],
            niit_threshold=_NIIT_THRESHOLD[fs],
            ctc_phaseout_start=_CTC_PHASEOUT_START[fs],
            enable_ctc=enable_ctc,
            enable_niit=enable_niit,
        )

    def reindexed(self, index_factor: float) -> "TaxParams":
        return TaxParams.for_year(self.filing_status, index_factor,
                                  self.enable_ctc, self.enable_niit)


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass
class TaxYearInput:
    """Everything taxable that happened during one household-year.

    All amounts are nominal dollars for the sim year the params belong to.
    """

    wages_by_adult: tuple[float, ...] = ()  # gross wages per adult (FICA base)
    pretax_401k: float = 0.0        # employee elective deferrals (both adults)
    hsa_contributions: float = 0.0  # above-the-line (payroll-tax interaction ignored)
    trad_taxable_withdrawals: float = 0.0  # taxable portion of trad 401k/IRA withdrawals (incl. RMDs)
    roth_conversion_taxable: float = 0.0   # taxable portion of conversions
    interest_ordinary: float = 0.0  # bond/cash interest in taxable + cash accounts
    qualified_dividends: float = 0.0
    realized_lt_gains: float = 0.0  # net realized LTG this year, may be negative
    other_ordinary_income: float = 0.0  # e.g. non-qualified 529 earnings
    ss_benefits_gross: float = 0.0  # total Social Security received
    penalty_base: float = 0.0       # amount subject to the 10% early penalty
    num_ctc_children: int = 0       # children under 17 at year end
    capital_loss_carryforward_in: float = 0.0  # positive number

    @property
    def total_wages(self) -> float:
        return sum(self.wages_by_adult)


@dataclass
class TaxResult:
    ordinary_taxable_income: float = 0.0
    preferential_income: float = 0.0   # QDI + net LTG taxed at cap-gains rates
    taxable_income: float = 0.0
    agi: float = 0.0
    magi: float = 0.0
    ss_taxable: float = 0.0
    ordinary_tax: float = 0.0
    ltcg_tax: float = 0.0
    ctc: float = 0.0
    income_tax: float = 0.0            # ordinary + ltcg − CTC + NIIT
    niit: float = 0.0
    payroll_tax: float = 0.0
    penalties: float = 0.0
    total_tax: float = 0.0             # income + payroll + penalties
    capital_loss_carryforward_out: float = 0.0
    marginal_ordinary_rate: float = 0.0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def tax_from_brackets(taxable: float, brackets: list[tuple[float, float]]) -> float:
    """Progressive tax on ``taxable`` using (top, rate) bracket rows."""
    if taxable <= 0:
        return 0.0
    tax = 0.0
    lower = 0.0
    for top, rate in brackets:
        if taxable <= lower:
            break
        span = min(taxable, top) - lower
        tax += span * rate
        lower = top
    return tax


def marginal_rate(taxable: float, brackets: list[tuple[float, float]]) -> float:
    for top, rate in brackets:
        if taxable < top:
            return rate
    return brackets[-1][1]


def taxable_social_security(ss_benefits: float, other_agi: float,
                            params: TaxParams) -> float:
    """Two-threshold provisional-income worksheet (IRS Pub 915, simplified).

    ``other_agi`` is AGI excluding Social Security (we model no tax-exempt
    interest). Returns the taxable portion, capped at 85% of benefits.
    """
    if ss_benefits <= 0:
        return 0.0
    provisional = other_agi + 0.5 * ss_benefits
    t1, t2 = params.ss_tax_t1, params.ss_tax_t2
    if provisional <= t1:
        return 0.0
    if provisional <= t2:
        return min(0.5 * (provisional - t1), 0.5 * ss_benefits)
    lower_piece = min(0.5 * (t2 - t1), 0.5 * ss_benefits)
    return min(0.85 * (provisional - t2) + lower_piece, 0.85 * ss_benefits)


def compute_taxes(inp: TaxYearInput, params: TaxParams) -> TaxResult:
    """Compute the household's federal tax for one year.

    Order of operations mirrors a Form 1040:
      1. net capital gains against losses + carryforward (Schedule D)
      2. ordinary income & AGI (with the taxable-SS worksheet)
      3. taxable income after the standard deduction
      4. bracket tax on the ordinary slice; 0/15/20% stacking on the
         preferential slice (QDI + net LTG)
      5. CTC, NIIT, payroll taxes, early-withdrawal penalties
    """
    res = TaxResult()

    # -- 1. capital gain/loss netting --------------------------------------
    net_gain = inp.realized_lt_gains - inp.capital_loss_carryforward_in
    ordinary_loss_offset = 0.0
    if net_gain < 0:
        ordinary_loss_offset = min(-net_gain, params.capital_loss_ordinary_offset)
        res.capital_loss_carryforward_out = -net_gain - ordinary_loss_offset
        net_gain = 0.0

    # -- 2. income aggregation ---------------------------------------------
    ordinary_income = (
        inp.total_wages
        - inp.pretax_401k
        - inp.hsa_contributions
        + inp.trad_taxable_withdrawals
        + inp.roth_conversion_taxable
        + inp.interest_ordinary
        + inp.other_ordinary_income
        - ordinary_loss_offset
    )
    preferential = inp.qualified_dividends + net_gain
    agi_ex_ss = ordinary_income + preferential

    res.ss_taxable = taxable_social_security(inp.ss_benefits_gross, agi_ex_ss, params)
    ordinary_income += res.ss_taxable
    res.agi = agi_ex_ss + res.ss_taxable
    res.magi = res.agi  # no foreign-income/muni add-backs modeled
    res.preferential_income = preferential

    # -- 3. taxable income ---------------------------------------------------
    res.taxable_income = max(0.0, res.agi - params.std_deduction)
    # the deduction soaks up ordinary income first (standard 1040 mechanics:
    # the preferential slice sits on top of the stack)
    pref_in_taxable = min(preferential, res.taxable_income)
    res.ordinary_taxable_income = res.taxable_income - pref_in_taxable

    # -- 4. bracket taxes ----------------------------------------------------
    res.ordinary_tax = tax_from_brackets(res.ordinary_taxable_income,
                                         params.ordinary_brackets)
    res.marginal_ordinary_rate = marginal_rate(res.ordinary_taxable_income,
                                               params.ordinary_brackets)

    # LTCG stacking: the preferential slice occupies
    # [ordinary_taxable, ordinary_taxable + pref] of taxable income and is
    # taxed at 0/15/20% based on where it falls relative to the LTCG breaks.
    lo = res.ordinary_taxable_income
    hi = lo + pref_in_taxable
    ltcg_tax = 0.0
    ltcg_tax += 0.00 * max(0.0, min(hi, params.ltcg_0_top) - lo)
    ltcg_tax += 0.15 * max(0.0, min(hi, params.ltcg_15_top) - max(lo, params.ltcg_0_top))
    ltcg_tax += params.ltcg_top_rate * max(0.0, hi - max(lo, params.ltcg_15_top))
    res.ltcg_tax = ltcg_tax

    # -- 5a. Child Tax Credit (nonrefundable simplification) -----------------
    gross_income_tax = res.ordinary_tax + res.ltcg_tax
    if params.enable_ctc and inp.num_ctc_children > 0:
        credit = params.ctc_per_child * inp.num_ctc_children
        over = max(0.0, res.magi - params.ctc_phaseout_start)
        if over > 0:
            # $50 per $1,000 (or fraction thereof) over the threshold
            import math
            credit -= 50.0 * math.ceil(over / 1000.0)
        res.ctc = max(0.0, min(credit, gross_income_tax))

    # -- 5b. NIIT -------------------------------------------------------------
    if params.enable_niit:
        nii = max(0.0, inp.interest_ordinary + inp.qualified_dividends + net_gain)
        res.niit = params.niit_rate * min(nii, max(0.0, res.magi - params.niit_threshold))

    res.income_tax = max(0.0, gross_income_tax - res.ctc) + res.niit

    # -- 5c. payroll (employee side) ------------------------------------------
    payroll = 0.0
    for w in inp.wages_by_adult:
        payroll += params.ss_rate * min(w, params.ss_wage_base)
        payroll += params.medicare_rate * w
    payroll += params.addl_medicare_rate * max(
        0.0, inp.total_wages - params.addl_medicare_threshold)
    res.payroll_tax = payroll

    # -- 5d. early-withdrawal penalties ---------------------------------------
    res.penalties = params.early_withdrawal_penalty_rate * max(0.0, inp.penalty_base)

    res.total_tax = res.income_tax + res.payroll_tax + res.penalties
    return res


def incremental_tax(base: TaxYearInput, params: TaxParams,
                    **overrides: float) -> float:
    """Total tax of (base with overrides) minus total tax of base.

    Utility for agents/baselines sizing Roth conversions against bracket tops.
    """
    modified = replace(base, **overrides)
    return compute_taxes(modified, params).total_tax - compute_taxes(base, params).total_tax


def ordinary_room_to_bracket_top(inp: TaxYearInput, params: TaxParams,
                                 bracket_rate: float) -> float:
    """How much more *ordinary* income fits before exceeding the top of the
    bracket with the given rate (0 if already past it).

    Used by the rule-based expert to size Roth conversions ("fill the 12%
    bracket"). Ignores knock-on effects on SS taxation (the caller converts
    before claiming SS in the intended use)."""
    res = compute_taxes(inp, params)
    top = None
    for t, r in params.ordinary_brackets:
        if abs(r - bracket_rate) < 1e-9:
            top = t
            break
    if top is None or top == float("inf"):
        return 0.0
    return max(0.0, top - res.ordinary_taxable_income)
