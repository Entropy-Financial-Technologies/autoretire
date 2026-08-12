"""LLM plumbing: JSON extraction, schema retry, hold-prior fallback, cache."""

import json

from finplan_arena.agents.base import Decision
from finplan_arena.agents.llm_agent import (CallableProvider, LLMAgent,
                                            LLMConfig, MockLLMAgent,
                                            PromptCache, extract_json_object)
from finplan_arena.core.simulator import SimConfig, build_observation
from finplan_arena.scenarios.library import get_scenario


def _obs():
    st = get_scenario("meridian").build_initial_state()
    return build_observation(st, SimConfig(), "meridian")


def _agent(fn, retries=2, cache_dir=None):
    cfg = LLMConfig(provider="callable", model="test", cache_dir=cache_dir,
                    max_schema_retries=retries, max_transport_retries=1,
                    transport_backoff_s=0.0)
    return LLMAgent(cfg, provider=CallableProvider(fn))


GOOD = json.dumps({
    "rationale": "steady as she goes",
    "annual_spending_discretionary": 20000,
    "contributions": {"401k_a1": 8700, "529_child1": 4000},
    "withdrawals": {},
    "allocations": {"taxable": {"stocks_us": 0.6, "stocks_intl": 0.2,
                                "bonds": 0.15, "cash": 0.05}},
    "retire_now": {"adult1": False, "adult2": False},
    "claim_social_security": {"adult1": False, "adult2": False},
    "tax_loss_harvest": False,
    "roth_conversion_amount": 0,
})


def test_extract_json_variants():
    assert json.loads(extract_json_object(GOOD))
    fenced = f"Here you go:\n```json\n{GOOD}\n```\nHope that helps!"
    assert json.loads(extract_json_object(fenced))
    prose = f"I think the right plan is {GOOD} — let me know."
    assert json.loads(extract_json_object(prose))
    nested = '{"a": {"b": "with } brace in string"}, "c": 1} trailing'
    assert json.loads(extract_json_object(nested)) == {
        "a": {"b": "with } brace in string"}, "c": 1}


def test_agent_parses_good_response():
    agent = _agent(lambda s, u: GOOD)
    d = agent.decide(_obs())
    assert isinstance(d, Decision)
    assert d.contributions.k401_a1 == 8700
    assert agent.llm_failures == 0


def test_agent_retries_with_error_feedback_then_succeeds():
    calls = []

    def flaky(system, user):
        calls.append(user)
        if len(calls) == 1:
            return '{"annual_spending_discretionary": "lots"}'  # type error
        return GOOD

    agent = _agent(flaky)
    d = agent.decide(_obs())
    assert d.contributions.k401_a1 == 8700
    assert len(calls) == 2
    assert "FAILED VALIDATION" in calls[1]      # error was fed back
    assert agent.llm_failures == 0


def test_agent_rejects_hallucinated_fields():
    bad = json.loads(GOOD)
    bad["buy_gold"] = True
    agent = _agent(lambda s, u: json.dumps(bad))
    agent.decide(_obs())
    assert agent.llm_failures == 1              # extra="forbid" did its job


def test_agent_falls_back_to_hold_prior_after_retries():
    agent = _agent(lambda s, u: "I would rather write an essay about bonds.")
    obs = _obs()
    obs.prior_decision = json.loads(GOOD)
    obs.state["prior_decision"] = json.loads(GOOD)
    d = agent.decide(obs)
    assert agent.llm_failures == 1
    assert "fallback" in d.rationale
    assert d.contributions.k401_a1 == 8700      # held prior contributions
    assert not d.claim_social_security.adult1   # one-shots cleared


def test_prompt_cache_roundtrip(tmp_path):
    calls = {"n": 0}

    def counting(system, user):
        calls["n"] += 1
        return GOOD

    agent = _agent(counting, cache_dir=str(tmp_path / "cache"))
    obs = _obs()
    agent.decide(obs)
    agent.decide(obs)                            # identical prompt → cache hit
    assert calls["n"] == 1
    key = PromptCache.key("m", 0.1, "s", "u")
    assert len(key) == 64


def test_mock_llm_exercises_full_pipeline():
    agent = MockLLMAgent()
    d = agent.decide(_obs())
    assert isinstance(d, Decision)
    assert d.contributions.k401_a1 > 0           # saves something
    assert agent.llm_failures == 0
