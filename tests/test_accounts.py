"""Account rules: limits, RMDs, SS claiming factors, withdrawal
characterization, penalties, backdoor pro-rata."""

import pytest

from finplan_arena.core.accounts import (ContributionLimits,
                                         backdoor_roth_contribution,
                                         employer_match,
                                         required_minimum_distribution,
                                         roth_convert, rmd_factor,
                                         ss_claiming_factor,
                                         withdraw_from_account)
from finplan_arena.core.state import Account, AccountKind, Allocation


def _acct(kind: AccountKind, balance: float, basis: float = 0.0,
          owner: str = "a1") -> Account:
    return Account(account_id="x", kind=kind, owner=owner, balance=balance,
                   basis=basis, alloc=Allocation(cash=1.0))


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_2025_limits_baseline():
    lim = ContributionLimits.for_year(1.0, "mfj")
    assert lim.max_401k_employee(40) == 23_500.0
    assert lim.max_401k_employee(50) == 31_000.0     # +7,500 catch-up
    assert lim.max_ira(40) == 7_000.0
    assert lim.max_ira(55) == 8_000.0
    assert lim.max_hsa([40, 41]) == 8_550.0
    assert lim.max_hsa([56, 41]) == 9_550.0
    assert lim.max_hsa([56, 57]) == 10_550.0
    assert lim.limit_529_per_child == 38_000.0        # 2 × $19k gift exclusion


def test_limits_index_with_inflation():
    lim = ContributionLimits.for_year(1.10, "mfj")
    assert lim.limit_401k_employee == pytest.approx(23_500 * 1.10)
    # the Roth phase-out is indexed in law, unlike the SS/NIIT thresholds
    assert lim.roth_magi_phaseout_lo == pytest.approx(236_000 * 1.10)


def test_roth_phaseout_fraction():
    lim = ContributionLimits.for_year(1.0, "mfj")
    assert lim.roth_ira_allowed_fraction(200_000) == 1.0
    assert lim.roth_ira_allowed_fraction(241_000) == pytest.approx(0.5)
    assert lim.roth_ira_allowed_fraction(250_000) == 0.0


def test_employer_match():
    # 50% of the first 6% of a $100k salary
    assert employer_match(3_000, 100_000, 0.5, 0.06) == pytest.approx(1_500)
    assert employer_match(10_000, 100_000, 0.5, 0.06) == pytest.approx(3_000)
    assert employer_match(10_000, 0.0, 0.5, 0.06) == 0.0


# ---------------------------------------------------------------------------
# RMDs
# ---------------------------------------------------------------------------


def test_rmd_math():
    assert required_minimum_distribution(72, 500_000) == 0.0
    assert required_minimum_distribution(73, 500_000) == pytest.approx(500_000 / 26.5)
    assert required_minimum_distribution(80, 1_000_000) == pytest.approx(1_000_000 / 20.2)
    assert required_minimum_distribution(95, 100_000) == pytest.approx(100_000 / 8.9)
    assert rmd_factor(120) > 0  # table clamps, never KeyErrors


# ---------------------------------------------------------------------------
# Social Security claiming
# ---------------------------------------------------------------------------


def test_ss_claiming_factors():
    assert ss_claiming_factor(62) == pytest.approx(0.70)
    assert ss_claiming_factor(65) == pytest.approx(0.88)
    assert ss_claiming_factor(67) == pytest.approx(1.00)
    assert ss_claiming_factor(70) == pytest.approx(1.24)
    assert ss_claiming_factor(75) == pytest.approx(1.24)  # clamped
    assert ss_claiming_factor(60) == pytest.approx(0.70)  # clamped


# ---------------------------------------------------------------------------
# Withdrawal characterization
# ---------------------------------------------------------------------------


def test_taxable_withdrawal_blended_basis():
    a = _acct(AccountKind.TAXABLE, 100_000, basis=75_000)
    eff = withdraw_from_account(a, 20_000, owner_age=45)
    assert eff.gross == 20_000
    assert eff.realized_lt_gain == pytest.approx(5_000)   # 25% gain slice
    assert eff.penalty_base == 0.0
    assert a.balance == pytest.approx(80_000)
    assert a.basis == pytest.approx(60_000)               # ratio preserved


def test_trad_401k_early_withdrawal_penalized():
    a = _acct(AccountKind.TRAD_401K, 50_000)
    eff = withdraw_from_account(a, 10_000, owner_age=45)
    assert eff.ordinary_income == 10_000
    assert eff.penalty_base == 10_000


def test_trad_401k_at_59_is_penalty_free():
    a = _acct(AccountKind.TRAD_401K, 50_000)
    eff = withdraw_from_account(a, 10_000, owner_age=59)  # hits 59.5 mid-year
    assert eff.penalty_base == 0.0


def test_ira_education_exception_skips_penalty():
    a = _acct(AccountKind.TRAD_IRA, 50_000)
    eff = withdraw_from_account(a, 10_000, owner_age=45, penalty_exempt=True)
    assert eff.ordinary_income == 10_000
    assert eff.penalty_base == 0.0


def test_roth_basis_first_then_earnings():
    a = _acct(AccountKind.ROTH_IRA, 30_000, basis=20_000)
    eff = withdraw_from_account(a, 25_000, owner_age=45)
    assert eff.tax_free == pytest.approx(20_000)
    assert eff.ordinary_income == pytest.approx(5_000)    # young earnings
    assert eff.penalty_base == pytest.approx(5_000)
    a2 = _acct(AccountKind.ROTH_IRA, 30_000, basis=20_000)
    eff2 = withdraw_from_account(a2, 25_000, owner_age=65)
    assert eff2.tax_free == pytest.approx(25_000)         # qualified
    assert eff2.penalty_base == 0.0


def test_529_nonqualified_taxes_earnings_only():
    a = _acct(AccountKind.COLLEGE_529, 40_000, basis=30_000, owner="child1")
    eff = withdraw_from_account(a, 10_000, owner_age=99)  # not qualified
    assert eff.ordinary_income == pytest.approx(2_500)    # earnings slice
    assert eff.penalty_base == pytest.approx(2_500)
    assert eff.tax_free == pytest.approx(7_500)
    q = _acct(AccountKind.COLLEGE_529, 40_000, basis=30_000, owner="child1")
    eq = withdraw_from_account(q, 10_000, owner_age=99, qualified_529=True)
    assert eq.tax_free == 10_000 and eq.ordinary_income == 0.0


def test_withdrawal_capped_at_balance():
    a = _acct(AccountKind.TAXABLE, 5_000, basis=5_000)
    eff = withdraw_from_account(a, 10_000, owner_age=70)
    assert eff.gross == 5_000
    assert a.balance == 0.0


# ---------------------------------------------------------------------------
# Conversions & backdoor
# ---------------------------------------------------------------------------


def test_roth_conversion_fully_taxable_no_penalty():
    src = _acct(AccountKind.TRAD_401K, 100_000)
    roth = _acct(AccountKind.ROTH_IRA, 10_000, basis=10_000)
    eff = roth_convert(src, roth, 30_000)
    assert eff.taxable_ordinary == 30_000
    assert roth.balance == 40_000
    assert roth.basis == 40_000     # converted dollars are basis


def test_backdoor_clean_when_no_pretax_ira():
    roth = _acct(AccountKind.ROTH_IRA, 0, basis=0)
    eff = backdoor_roth_contribution(None, roth, 7_000)
    assert eff.taxable_ordinary == 0.0
    assert roth.balance == 7_000 and roth.basis == 7_000


def test_backdoor_pro_rata_with_pretax_ira():
    """$93k pre-tax IRA + $7k backdoor: nontaxable fraction is
    7,000/100,000 = 7%, so $6,510 of the conversion is taxable and $6,510
    of basis stays stranded in the IRA (Form 8606 outcome)."""
    trad = _acct(AccountKind.TRAD_IRA, 93_000, basis=0)
    roth = _acct(AccountKind.ROTH_IRA, 0, basis=0)
    eff = backdoor_roth_contribution(trad, roth, 7_000)
    assert eff.taxable_ordinary == pytest.approx(6_510.0)
    assert roth.balance == 7_000
    assert trad.balance == pytest.approx(93_000)
    assert trad.basis == pytest.approx(6_510.0)
