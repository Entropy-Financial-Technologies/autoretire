"""Decision validator/clipper: every infeasible proposal must be clipped to
the nearest feasible value AND recorded as a violation."""

import pytest

from finplan_arena.agents.base import Decision
from finplan_arena.agents.validation import validate_decision
from finplan_arena.core.accounts import ContributionLimits
from finplan_arena.scenarios.library import get_scenario


def _setup(name="meridian"):
    st = get_scenario(name).build_initial_state()
    limits = ContributionLimits.for_year(st.inflation_index, st.filing_status)
    return st, limits


def _codes(violations):
    return {v.code for v in violations}


def test_401k_over_limit_clipped():
    st, lim = _setup()
    d = Decision()
    d.contributions.k401_a1 = 60_000.0
    vetted, v = validate_decision(d, st, lim, estimated_magi=200_000)
    assert vetted.contributions.k401_a1 == pytest.approx(23_500.0)
    assert "401k_limit" in _codes(v)


def test_401k_requires_wages():
    st, lim = _setup()
    st.adults[0].retired = True
    d = Decision()
    d.contributions.k401_a1 = 10_000.0
    vetted, v = validate_decision(d, st, lim, 100_000)
    assert vetted.contributions.k401_a1 == 0.0
    assert "not_working" in _codes(v)


def test_roth_phaseout_blocks_direct_but_not_backdoor():
    st, lim = _setup()
    d = Decision()
    d.contributions.roth_ira_a1 = 7_000.0
    vetted, v = validate_decision(d, st, lim, estimated_magi=300_000)
    assert vetted.contributions.roth_ira_a1 == 0.0
    assert "roth_magi_phaseout" in _codes(v)

    d2 = Decision()
    d2.contributions.roth_ira_a1 = 7_000.0
    d2.contributions.backdoor_roth_a1 = True
    vetted2, v2 = validate_decision(d2, st, lim, estimated_magi=300_000)
    assert vetted2.contributions.roth_ira_a1 == pytest.approx(7_000.0)
    assert "roth_magi_phaseout" not in _codes(v2)


def test_spending_clipped_to_bounds():
    st, lim = _setup()
    d = Decision(annual_spending_discretionary=10_000_000.0)
    vetted, v = validate_decision(d, st, lim, 200_000)
    assert vetted.annual_spending_discretionary == st.discretionary_ceiling
    assert "spending_bounds" in _codes(v)


def test_withdrawal_beyond_balance_clipped():
    st, lim = _setup()
    d = Decision()
    d.withdrawals.taxable = 10_000_000.0
    vetted, v = validate_decision(d, st, lim, 200_000)
    assert vetted.withdrawals.taxable == pytest.approx(
        st.accounts["taxable"].balance)
    assert "withdrawal_exceeds_balance" in _codes(v)


def test_allocation_renormalized_and_flagged():
    st, lim = _setup()
    d = Decision(allocations={"taxable": {"stocks_us": 0.9, "bonds": 0.3}})
    vetted, v = validate_decision(d, st, lim, 200_000)
    w = vetted.allocations["taxable"]
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["stocks_us"] == pytest.approx(0.75)
    assert "allocation_sum" in _codes(v)


def test_unknown_account_allocation_dropped():
    st, lim = _setup()
    d = Decision(allocations={"crypto_wallet": {"stocks_us": 1.0}})
    vetted, v = validate_decision(d, st, lim, 200_000)
    assert "crypto_wallet" not in vetted.allocations
    assert "unknown_account" in _codes(v)


def test_ss_claim_too_young_rejected():
    st, lim = _setup()
    d = Decision()
    d.claim_social_security.adult1 = True   # age 38
    vetted, v = validate_decision(d, st, lim, 200_000)
    assert not vetted.claim_social_security.adult1
    assert "ss_too_young" in _codes(v)


def test_single_parent_has_no_adult2():
    st, lim = _setup("harbor")
    d = Decision()
    d.contributions.k401_a2 = 5_000.0
    d.claim_social_security.adult2 = True
    vetted, v = validate_decision(d, st, lim, 60_000)
    assert vetted.contributions.k401_a2 == 0.0
    assert not vetted.claim_social_security.adult2
    assert "no_such_adult" in _codes(v)


def test_hsa_rejected_when_disabled():
    st, lim = _setup("meridian")  # meridian has no HDHP
    d = Decision()
    d.contributions.hsa = 5_000.0
    vetted, v = validate_decision(d, st, lim, 200_000)
    assert vetted.contributions.hsa == 0.0
    assert "hsa_not_available" in _codes(v)


def test_conversion_capped_at_pretax_balance():
    st, lim = _setup()
    total_pretax = 270_000.0  # meridian 401ks
    d = Decision(roth_conversion_amount=1_000_000.0)
    vetted, v = validate_decision(d, st, lim, 200_000)
    assert vetted.roth_conversion_amount == pytest.approx(total_pretax)
    assert "conversion_exceeds_balance" in _codes(v)


def test_feasible_decision_produces_no_violations():
    st, lim = _setup()
    d = Decision()
    d.contributions.k401_a1 = 10_000.0
    d.contributions.c529_child1 = 5_000.0
    d.annual_spending_discretionary = 20_000.0
    _vetted, v = validate_decision(d, st, lim, 180_000)
    assert v == []
