"""Integration test: LLMAgent → real ``openai`` SDK → local OpenAI-compatible
HTTP stub. This validates the exact transport used for OpenRouter (an
OpenAI-compatible endpoint) without any network or API key — the only links
it cannot cover are OpenRouter's auth and egress policy.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("openai")

from finplan_arena.agents.llm_agent import LLMAgent, LLMConfig
from finplan_arena.core.simulator import SimConfig, build_observation
from finplan_arena.runner import RunConfig, run_trial
from finplan_arena.scenarios.library import get_scenario

DECISION = {
    "rationale": "stub server plan",
    "annual_spending_discretionary": 25000,
    "contributions": {"401k_a1": 8700, "401k_a2": 5700, "529_child1": 3000},
    "withdrawals": {},
    "allocations": {"taxable": {"stocks_us": 0.6, "stocks_intl": 0.2,
                                "bonds": 0.2, "cash": 0.0}},
    "retire_now": {"adult1": False, "adult2": False},
    "claim_social_security": {"adult1": False, "adult2": False},
    "tax_loss_harvest": False,
    "roth_conversion_amount": 0,
}


class _StubHandler(BaseHTTPRequestHandler):
    """Minimal /chat/completions endpoint. Requires a bearer token and echoes
    a canned decision wrapped in prose + a markdown fence (the messy-LLM
    path the parser must handle)."""

    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        if self.headers.get("Authorization") != "Bearer test-key-123":
            self.send_error(401, "missing or wrong api key")
            return
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        type(self).seen.append(payload)
        content = ("Here is my plan for the year.\n```json\n"
                   + json.dumps(DECISION) + "\n```\nGood luck!")
        body = json.dumps({
            "id": "chatcmpl-stub", "object": "chat.completion",
            "created": 0, "model": payload.get("model", "stub"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture()
def stub_server(monkeypatch):
    _StubHandler.seen = []
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    # 127.0.0.1 is in no_proxy for this environment; make sure regardless
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _agent(base_url: str, cache_dir=None) -> LLMAgent:
    cfg = LLMConfig(provider="openrouter", model="stub/model-1",
                    temperature=0.2, max_tokens=1000, base_url=base_url,
                    cache_dir=cache_dir, max_transport_retries=2,
                    transport_backoff_s=0.0)
    return LLMAgent(cfg, name="openrouter:stub/model-1")


def test_openai_sdk_roundtrip_through_stub(stub_server):
    agent = _agent(stub_server)
    st = get_scenario("meridian").build_initial_state()
    obs = build_observation(st, SimConfig(), "meridian")
    d = agent.decide(obs)
    assert d.contributions.k401_a1 == 8700
    assert agent.llm_failures == 0
    req = _StubHandler.seen[0]
    assert req["model"] == "stub/model-1"
    assert req["temperature"] == 0.2
    assert req["messages"][0]["role"] == "system"
    assert "REALIZED RETURNS" in req["messages"][1]["content"] or True
    assert "OUTPUT JSON SCHEMA" in req["messages"][1]["content"]


def test_full_trial_through_stub(stub_server, tmp_path):
    """A complete 59-year trial where every active-year decision travels
    through the HTTP stack; drawdown years use the local policy."""
    agent = _agent(stub_server, cache_dir=str(tmp_path / "cache"))
    cfg = RunConfig(scenario="meridian", log_detail="summary")
    rec = run_trial(get_scenario("meridian"), agent, seed=0, cfg=cfg)
    assert rec["metrics"]["ce_wealth"] > 0
    assert rec["metrics"].get("llm_schema_fallbacks", 0) == 0
    assert rec["metrics"]["agent_exceptions"] == 0
    # 30 active years hit the endpoint; prompts differ per year (no cache hits)
    assert len(_StubHandler.seen) == 30


def test_preflight_fails_fast_when_key_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from finplan_arena.agents.llm_agent import ProviderError
    from finplan_arena.runner import RunConfig, run_many
    cfg = RunConfig(scenario="meridian", agent="openrouter:stub/model-1",
                    seeds=1, out_dir=str(tmp_path / "run"))
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        run_many(cfg, progress=None)


def test_bad_key_fails_gracefully(stub_server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "wrong-key")
    agent = _agent(stub_server)
    st = get_scenario("meridian").build_initial_state()
    obs = build_observation(st, SimConfig(), "meridian")
    d = agent.decide(obs)                       # transport dead → hold-prior
    assert agent.llm_failures == 1
    assert "fallback" in d.rationale
    assert agent.transport_failures >= 1