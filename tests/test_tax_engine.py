"""Tax engine vs. hand-computed returns (2025 law, MFJ + one HoH case).

Every expected number below was worked by hand from the embedded 2025
parameters (brackets per Rev. Proc. 2024-40). If a constant in
``tax_engine.py`` changes, these change with it — deliberately: the tests
pin the arithmetic, the constants pin the law.
"""

import pytest

from finplan_arena.core.tax_engine import (TaxParams, TaxYearInput,
                                           compute_taxes,
                                           taxable_social_security,
                                           tax_from_brackets)

MFJ = TaxParams.for_year("mfj", 1.0)
HOH = TaxParams.for_year("hoh", 1.0)


def test_case1_wage_couple_with_kids():
    """W-2 couple, $240k wages, $14.4k 401(k), 2 CTC kids.

    ordinary = 240,000 − 14,400 = 225,600 → taxable 195,600
    tax = 2,385 + 8,772 + 22%×98,650 = 32,860 → −4,000 CTC = 28,860
    payroll = (8,990 + 2,102.50) + (5,890 + 1,377.50) = 18,360
    """
    inp = TaxYearInput(wages_by_adult=(145_000.0, 95_000.0),
                       pretax_401k=14_400.0, num_ctc_children=2)
    r = compute_taxes(inp, MFJ)
    assert r.agi == pytest.approx(225_600.0)
    assert r.taxable_income == pytest.approx(195_600.0)
    assert r.ordinary_tax == pytest.approx(32_860.0)
    assert r.ctc == pytest.approx(4_000.0)
    assert r.income_tax == pytest.approx(28_860.0)
    assert r.payroll_tax == pytest.approx(18_360.0)
    assert r.niit == 0.0
    assert r.penalties == 0.0
    assert r.total_tax == pytest.approx(47_220.0)


def test_case2_retirees_ss_ltcg_stacking():
    """Retirees: $60k SS, $40k trad withdrawals, $5k QDI, $20k LTG.

    provisional = 65,000 + 30,000 = 95,000 → taxable SS
      = min(0.85×51,000 + min(6,000, 30,000), 51,000) = 49,350
    AGI = 114,350 → taxable 84,350; pref 25,000 → ordinary slice 59,350
    ordinary tax = 2,385 + 12%×35,500 = 6,645
    LTCG: 59,350→84,350 all under the 96,700 0% top → $0 (the classic
    0%-bracket harvest window the engine must reproduce)
    """
    inp = TaxYearInput(trad_taxable_withdrawals=40_000.0,
                       qualified_dividends=5_000.0,
                       realized_lt_gains=20_000.0,
                       ss_benefits_gross=60_000.0)
    r = compute_taxes(inp, MFJ)
    assert r.ss_taxable == pytest.approx(49_350.0)
    assert r.agi == pytest.approx(114_350.0)
    assert r.taxable_income == pytest.approx(84_350.0)
    assert r.ordinary_taxable_income == pytest.approx(59_350.0)
    assert r.ordinary_tax == pytest.approx(6_645.0)
    assert r.ltcg_tax == pytest.approx(0.0)
    assert r.total_tax == pytest.approx(6_645.0)


def test_case3_high_income_conversion_niit_addl_medicare():
    """$500k wages, $47k 401(k), $100k Roth conversion, $15k QDI,
    $10k interest, $50k LTG, 1 kid (CTC fully phased out).

    ordinary = 500,000 − 47,000 + 100,000 + 10,000 = 563,000; pref 65,000
    AGI 628,000 → taxable 598,000 → ordinary slice 533,000
    ordinary tax = 2,385 + 8,772 + 24,145 + 45,096 + 34,064 + 35%×31,950
                 = 125,644.50
    LTCG: all 65,000 in the 15% band (below 600,050) = 9,750
    NIIT = 3.8% × min(75,000, 378,000) = 2,850
    payroll = 10,918.20+4,640 + 10,918.20+2,610 + 0.9%×250,000 = 31,336.40
    """
    inp = TaxYearInput(wages_by_adult=(320_000.0, 180_000.0),
                       pretax_401k=47_000.0,
                       roth_conversion_taxable=100_000.0,
                       qualified_dividends=15_000.0,
                       interest_ordinary=10_000.0,
                       realized_lt_gains=50_000.0,
                       num_ctc_children=1)
    r = compute_taxes(inp, MFJ)
    assert r.agi == pytest.approx(628_000.0)
    assert r.taxable_income == pytest.approx(598_000.0)
    assert r.ordinary_tax == pytest.approx(125_644.50)
    assert r.ltcg_tax == pytest.approx(9_750.0)
    assert r.ctc == 0.0
    assert r.niit == pytest.approx(2_850.0)
    assert r.income_tax == pytest.approx(138_244.50)
    assert r.payroll_tax == pytest.approx(31_336.40)
    assert r.total_tax == pytest.approx(169_580.90)


def test_case4_penalty_and_loss_carryforward():
    """$80k wages, $20k early trad withdrawal (age 45), $12k harvested loss.

    loss: 3,000 offsets ordinary, 9,000 carries forward
    ordinary = 80,000 + 20,000 − 3,000 = 97,000 → taxable 67,000
    tax = 2,385 + 12%×43,150 = 7,563; penalty = 2,000; payroll = 6,120
    """
    inp = TaxYearInput(wages_by_adult=(80_000.0, 0.0),
                       trad_taxable_withdrawals=20_000.0,
                       realized_lt_gains=-12_000.0,
                       penalty_base=20_000.0)
    r = compute_taxes(inp, MFJ)
    assert r.capital_loss_carryforward_out == pytest.approx(9_000.0)
    assert r.taxable_income == pytest.approx(67_000.0)
    assert r.income_tax == pytest.approx(7_563.0)
    assert r.penalties == pytest.approx(2_000.0)
    assert r.payroll_tax == pytest.approx(6_120.0)
    assert r.total_tax == pytest.approx(15_683.0)


def test_case5_head_of_household():
    """Single parent, $78k wages, 6% 401(k), 1 CTC kid.

    ordinary = 73,320 → taxable 50,820 (HoH deduction 22,500)
    tax = 1,700 + 12%×33,820 = 5,758.40 → −2,000 CTC = 3,758.40
    payroll = 5,967
    """
    inp = TaxYearInput(wages_by_adult=(78_000.0,), pretax_401k=4_680.0,
                       num_ctc_children=1)
    r = compute_taxes(inp, HOH)
    assert r.taxable_income == pytest.approx(50_820.0)
    assert r.income_tax == pytest.approx(3_758.40)
    assert r.payroll_tax == pytest.approx(5_967.0)
    assert r.total_tax == pytest.approx(9_725.40)


# ---------------------------------------------------------------------------
# Component-level checks
# ---------------------------------------------------------------------------


def test_ss_taxation_tiers():
    # below first threshold → nothing taxable
    assert taxable_social_security(20_000, 15_000, MFJ) == 0.0
    # middle tier: PI = 25,000 + 10,000 = 35,000 → 0.5×3,000 = 1,500
    assert taxable_social_security(20_000, 25_000, MFJ) == pytest.approx(1_500.0)
    # far above → capped at 85%
    assert taxable_social_security(40_000, 500_000, MFJ) == pytest.approx(34_000.0)


def test_deduction_soaks_ordinary_then_gains():
    # only LTG income, below the deduction+0% bracket → zero tax
    inp = TaxYearInput(realized_lt_gains=25_000.0)
    r = compute_taxes(inp, MFJ)
    assert r.taxable_income == pytest.approx(0.0)
    assert r.total_tax == 0.0


def test_loss_carryforward_consumed_by_later_gains():
    inp = TaxYearInput(realized_lt_gains=6_000.0,
                       capital_loss_carryforward_in=9_000.0)
    r = compute_taxes(inp, MFJ)
    # 6k gain absorbed, 3k of the remainder offsets ordinary, 0 carries on
    assert r.capital_loss_carryforward_out == pytest.approx(0.0)
    assert r.agi == pytest.approx(-3_000.0)


def test_bracket_indexing_scales_tax_down():
    """Same nominal income against inflated brackets → less tax (bracket
    creep in reverse: the engine indexes correctly)."""
    inp = TaxYearInput(wages_by_adult=(150_000.0, 90_000.0))
    base = compute_taxes(inp, MFJ).income_tax
    indexed = compute_taxes(inp, TaxParams.for_year("mfj", 1.30)).income_tax
    assert indexed < base


def test_ss_thresholds_do_not_index():
    p = TaxParams.for_year("mfj", 2.0)
    assert p.ss_tax_t1 == 32_000.0 and p.ss_tax_t2 == 44_000.0
    assert p.niit_threshold == 250_000.0
    assert p.ctc_phaseout_start == 400_000.0


def test_brackets_monotonic_and_continuous():
    xs = [0, 10_000, 23_850, 50_000, 96_950, 150_000, 206_700, 300_000,
          394_600, 450_000, 501_050, 700_000, 751_600, 1_000_000]
    taxes = [tax_from_brackets(x, MFJ.ordinary_brackets) for x in xs]
    assert all(b >= a for a, b in zip(taxes, taxes[1:]))
    # marginal never exceeds 37%
    for a, b, xa, xb in zip(taxes, taxes[1:], xs, xs[1:]):
        assert (b - a) <= 0.37 * (xb - xa) + 1e-9
