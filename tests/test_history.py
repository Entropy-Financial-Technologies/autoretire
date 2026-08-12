"""Agent-facing history: per-year digests accumulate in state, reach the
prompt (text + machine JSON) when enabled, and disappear with --no-history."""

import json

from finplan_arena.agents.base import Decision
from finplan_arena.core.simulator import (SimConfig, build_observation,
                                          history_record, simulate_year)
from finplan_arena.core.state import HouseholdState
from finplan_arena.markets.returns import make_generator
from finplan_arena.runner import RunConfig, load_trial, run_many
from finplan_arena.scenarios.library import get_scenario


def _one_year_log():
    scenario = get_scenario("meridian")
    state = scenario.build_initial_state()
    path = make_generator("block_bootstrap").generate_path(seed=0, n_years=1)
    decision = Decision(rationale="test year",
                        annual_spending_discretionary=20_000.0,
                        roth_conversion_amount=5_000.0)
    return simulate_year(state, decision, path[0], SimConfig())


def test_history_record_digest():
    _, log = _one_year_log()
    rec = history_record(log)
    assert rec["year"] == 0
    assert rec["rationale"] == "test year"
    assert rec["decision"]["spending"] == 20_000.0
    assert rec["decision"]["roth_conversion"] == 5_000.0
    # zero-valued decision fields are omitted from the digest
    assert "withdrawals" not in rec["decision"]
    assert "consumption_real" in rec["outcome"]
    assert "total_tax" in rec["outcome"]


def test_history_reaches_prompt_and_toggle():
    state, log = _one_year_log()
    state.history.append(history_record(log))

    obs = build_observation(state, SimConfig(), "meridian")
    text = obs.render_text()
    assert "YOUR FULL HISTORY" in text
    assert "test year" in text
    assert "history" in obs.state
    # survives the state round-trip agents use
    assert HouseholdState.from_dict(obs.state).history[0]["rationale"] == "test year"

    obs_off = build_observation(state, SimConfig(include_history=False),
                                "meridian")
    assert "YOUR FULL HISTORY" not in obs_off.render_text()
    assert "history" not in obs_off.state


def test_history_accumulates_in_trial(tmp_path):
    out = str(tmp_path / "run_hist")
    cfg = RunConfig(scenario="meridian", agent="naive", seeds=1, workers=1,
                    out_dir=out, log_detail="summary")
    run_many(cfg, progress=None)
    rec = load_trial(out, 0)
    hist = rec["final_state"]["history"]
    assert len(hist) == rec["years_simulated"]
    assert [h["year"] for h in hist] == list(range(rec["years_simulated"]))
    # digests are JSON-serializable and carry outcomes for every year
    json.dumps(hist)
    assert all("outcome" in h and "decision" in h for h in hist)
