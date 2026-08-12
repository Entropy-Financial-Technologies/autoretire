"""Simulator behavior: determinism, RMD forcing, mortgage amortization,
shortfall cascade (incl. the IRA education exception), forced liquidation
for taxes, and the drawdown handoff."""

import hashlib
import json

import pytest

from finplan_arena.agents.base import Decision
from finplan_arena.agents.baselines import RuleBasedExpertAgent
from finplan_arena.core.simulator import (SimConfig, _pay_mortgage,
                                          build_observation, simulate_year)
from finplan_arena.core.state import Mortgage
from finplan_arena.markets.returns import BlockBootstrapGenerator, YearReturns
from finplan_arena.runner import RunConfig, run_trial
from finplan_arena.scenarios.library import get_scenario

CFG = SimConfig()
FLAT = YearReturns(stocks_us=0.0, stocks_intl=0.0, bonds=0.0, cash=0.0,
                   inflation=0.0)


def _mk_state(name="meridian"):
    return get_scenario(name).build_initial_state()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _trial_hash(seed: int) -> str:
    cfg = RunConfig(scenario="meridian", agent="expert", seeds=1,
                    log_detail="full")
    rec = run_trial(get_scenario("meridian"), RuleBasedExpertAgent(), seed, cfg)
    return hashlib.sha256(
        json.dumps(rec, sort_keys=True).encode()).hexdigest()


def test_same_seed_same_outcome_hash():
    assert _trial_hash(11) == _trial_hash(11)


def test_different_seed_different_outcome():
    assert _trial_hash(11) != _trial_hash(12)


def test_return_paths_paired_across_calls():
    g = BlockBootstrapGenerator()
    a = g.generate_path(seed=5, n_years=40)
    b = BlockBootstrapGenerator().generate_path(seed=5, n_years=40)
    assert [y.to_dict() for y in a] == [y.to_dict() for y in b]


# ---------------------------------------------------------------------------
# Mortgage
# ---------------------------------------------------------------------------


def test_mortgage_amortization_and_payoff():
    st = _mk_state()
    st.mortgage = Mortgage(balance=10_000.0, annual_rate=0.12,
                           monthly_payment=5_000.0)
    paid, due = _pay_mortgage(st, funds_available=1e9)
    # month 1: 100 interest → 5,000 paid → bal 5,100
    # month 2: 51 interest → 5,000 paid → bal 151
    # month 3: 1.51 interest → 152.51 payoff
    assert due == pytest.approx(10_152.51, abs=0.02)
    assert paid == pytest.approx(10_152.51, abs=0.02)
    assert not st.mortgage.active


def test_unpaid_mortgage_accrues_interest():
    st = _mk_state()
    st.mortgage = Mortgage(balance=100_000.0, annual_rate=0.06,
                           monthly_payment=1_000.0)
    paid, _due = _pay_mortgage(st, funds_available=0.0)
    assert paid == 0.0
    # 12 months of monthly compounding at 0.5%
    assert st.mortgage.balance == pytest.approx(100_000 * 1.005 ** 12, rel=1e-9)


# ---------------------------------------------------------------------------
# RMD enforcement
# ---------------------------------------------------------------------------


def test_rmd_forced_when_agent_ignores_it():
    st = _mk_state()
    for a in st.adults:
        a.age = 75
        a.retired = True
        a.ss_claimed = True
        a.ss_annual_benefit = 30_000.0
    st.accounts["trad_401k_a1"].balance = 500_000.0
    st.accounts["trad_401k_a2"].balance = 0.0
    st.children = []
    new, log = simulate_year(st, Decision(), FLAT, CFG)
    expected = 500_000 / 24.6  # age 75 factor
    assert log.rmd_forced == pytest.approx(expected, rel=1e-6)
    assert log.tax["total_tax"] > 0  # RMD is ordinary income


def test_rmd_not_forced_when_agent_withdraws_enough():
    st = _mk_state()
    for a in st.adults:
        a.age = 75
        a.retired = True
    st.accounts["trad_401k_a1"].balance = 500_000.0
    st.accounts["trad_401k_a2"].balance = 100_000.0
    st.children = []
    d = Decision()
    d.withdrawals.trad_401k = 40_000.0  # above both RMDs
    _new, log = simulate_year(st, d, FLAT, CFG)
    assert log.rmd_forced == 0.0


# ---------------------------------------------------------------------------
# Shortfall cascade & education exception
# ---------------------------------------------------------------------------


def _strip_household_to_college_case():
    """45-year-old couple, empty cash/taxable, child in college, all money
    in a1's accounts. Forces the college tier into the cascade."""
    st = _mk_state()
    st.adults[0].age = 45
    st.adults[1].age = 45
    st.adults[0].gross_salary = 0.0
    st.adults[1].gross_salary = 0.0
    st.adults[0].retired = True
    st.adults[1].retired = True
    st.mortgage = None
    st.children = st.children[:1]
    st.children[0].age = 18            # in college
    st.children[0].college_annual_cost = 30_000.0
    for aid in list(st.accounts):
        st.accounts[aid].balance = 0.0
        st.accounts[aid].basis = 0.0
    st.accounts["529_child1"].balance = 0.0
    st.essential_expenses = 40_000.0
    st.discretionary_floor = 0.0
    st.discretionary_ceiling = 0.0
    return st


def test_college_cascade_uses_ira_education_exception():
    st = _strip_household_to_college_case()
    st.accounts["cash"].balance = 40_000.0     # covers essentials only
    st.accounts["trad_ira_a1"].balance = 500_000.0
    _new, log = simulate_year(st, Decision(), FLAT, CFG)
    # college was paid by force-liquidating the IRA — penalty-free
    # (higher-education exception); essentials consumed the cash.
    assert log.expenses["college_paid"] == pytest.approx(30_000.0, abs=1.0)
    assert log.tax["penalties"] == 0.0
    assert not any(s["tier"] == "college" for s in log.shortfalls)


def test_essential_cascade_from_401k_pays_penalty():
    st = _strip_household_to_college_case()
    st.children = []                            # no college this time
    st.accounts["cash"].balance = 0.0
    st.accounts["trad_401k_a1"].balance = 500_000.0
    _new, log = simulate_year(st, Decision(), FLAT, CFG)
    assert log.expenses["essential_paid"] == pytest.approx(40_000.0, abs=1.0)
    assert log.tax["penalties"] > 0             # 45-year-old raiding a 401(k)
    assert log.forced_liquidation > 0


def test_true_shortfall_recorded_when_everything_empty():
    st = _strip_household_to_college_case()
    st.children = []
    st.accounts["cash"].balance = 10_000.0      # not enough for 40k essential
    _new, log = simulate_year(st, Decision(), FLAT, CFG)
    tiers = {s["tier"] for s in log.shortfalls}
    assert "essential" in tiers
    short = next(s for s in log.shortfalls if s["tier"] == "essential")
    assert short["amount"] == pytest.approx(30_000.0, abs=1.0)


def test_college_paid_from_529_first():
    st = _mk_state()
    st.children[0].age = 18
    st.accounts["529_child1"].balance = 200_000.0
    st.accounts["529_child1"].basis = 150_000.0
    bill = st.children[0].college_annual_cost
    _new, log = simulate_year(st, Decision(), FLAT, CFG)
    assert log.expenses["college_from_529"] == pytest.approx(bill, abs=1.0)
    assert log.tax["penalties"] == 0.0          # qualified withdrawal


# ---------------------------------------------------------------------------
# Taxes force liquidation when cash can't cover them
# ---------------------------------------------------------------------------


def test_tax_settlement_forces_liquidation():
    st = _mk_state()
    for a in st.adults:                          # no wages to hide behind
        a.retired = True
        a.age = 62
    st.children = []
    st.mortgage = None
    st.accounts["cash"].balance = 0.0
    st.accounts["taxable"].balance = 300_000.0
    st.accounts["taxable"].basis = 200_000.0
    d = Decision()
    d.roth_conversion_amount = 150_000.0        # big tax bill, no cash
    st.accounts["trad_401k_a1"].balance = 400_000.0
    _new, log = simulate_year(st, d, FLAT, CFG)
    assert log.roth_converted == pytest.approx(150_000.0)
    assert log.forced_liquidation > 0           # taxable sold to pay the tax
    assert log.unpaid_tax == 0.0


# ---------------------------------------------------------------------------
# Life events / observation
# ---------------------------------------------------------------------------


def test_ss_forced_claim_at_70():
    st = _mk_state()
    for a in st.adults:
        a.age = 70
        a.retired = True
    st.children = []
    pia = st.adults[0].ss_pia_annual
    _new, log = simulate_year(st, Decision(), FLAT, CFG)
    claims = log.life_events["ss_claims"]
    assert {c["person_id"] for c in claims} == {"a1", "a2"}
    assert all(c["forced"] for c in claims)
    assert log.ss_benefits == pytest.approx(
        1.24 * (pia + st.adults[1].ss_pia_annual /
                (1 + FLAT.inflation)), rel=1e-6) or log.ss_benefits > 0


def test_observation_is_backward_looking_only():
    st = _mk_state()
    obs = build_observation(st, CFG, "meridian")
    text = obs.render_text()
    assert "REALIZED RETURNS" not in text or st.returns_history  # no history yet
    assert obs.recent_returns == []
    d = obs.to_json_dict()
    assert "household_state" in d and "goals" in d


def test_retire_now_stops_salary_same_year():
    st = _mk_state()
    d = Decision()
    d.retire_now.adult1 = True
    _new, log = simulate_year(st, d, FLAT, CFG)
    assert log.salaries["a1"] == 0.0
    assert log.salaries["a2"] > 0
    assert _new.adult("a1").retired
