"""LLM-backed agent.

Any OpenAI-compatible or Anthropic API endpoint works; model name,
temperature, and system prompt are config. Calls are cached on disk keyed by
(model, temperature, prompt hash) so reruns are cheap and resumable runs
don't re-bill.

Robustness contract (spec §5): on schema failure the agent retries up to
``max_schema_retries`` times with the validation error appended to the
prompt; after that it falls back to "hold prior decision" and records the
failure (``llm_failures`` on the agent, and the fallback rationale makes it
visible in trial logs).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import ValidationError

from ..core.state import HouseholdState
from .base import BaseAgent, Decision, Observation, decision_json_schema

DEFAULT_SYSTEM_PROMPT = """\
You are a meticulous household financial planner making ONE year of decisions
for a family, once per year, over a multi-decade horizon. You will see the
household's full state, its goals in priority order, this year's contribution
limits, recent realized market returns (never forward-looking information),
and your own prior-year decision and rationale.

Optimize for the household's stated goals in priority order; the headline
metric is certainty-equivalent lifetime consumption (CRRA), so smooth,
sustainable spending and avoiding ruin matter more than maximizing terminal
wealth. Mind taxes: asset location, Roth vs traditional timing, conversion
windows, capital-gains brackets, and early-withdrawal penalties all matter.
Avoid thrashing allocations year to year without cause.

Respond with ONLY a single JSON object matching the provided schema — no
markdown fences, no commentary outside the JSON. Keep "rationale" under 60
words; it is fed back to you next year for continuity."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class ProviderError(RuntimeError):
    pass


@dataclass
class AnthropicProvider:
    """Anthropic Messages API (lazy import; pip install anthropic)."""

    model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    base_url: Optional[str] = None
    api_key_env: str = "ANTHROPIC_API_KEY"

    def complete(self, system: str, user: str) -> str:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ProviderError("pip install anthropic (or use finplan-arena[llm])") from e
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        api_key = os.environ.get(self.api_key_env)
        client = anthropic.Anthropic(api_key=api_key, **kwargs)
        msg = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature, system=system,
            messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


@dataclass
class OpenAICompatProvider:
    """OpenAI SDK against api.openai.com or any compatible ``base_url``
    (OpenRouter, vLLM, LM Studio, ...).

    ``temperature=None`` omits the parameter entirely — some models
    (gpt-5/o-series) reject explicit temperatures."""

    model: str
    temperature: Optional[float] = 0.2
    max_tokens: int = 2000
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"

    def complete(self, system: str, user: str) -> str:
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ProviderError("pip install openai (or use finplan-arena[llm])") from e
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(
                f"environment variable {self.api_key_env} is not set")
        client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
        kwargs = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **kwargs)
        return resp.choices[0].message.content or ""


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class CallableProvider:
    """Wraps any (system, user) -> str callable. Used by tests to fake an
    LLM endpoint (canned/garbled responses) through the real parsing path."""

    fn: Callable[[str, str], str]
    model: str = "callable"
    temperature: float = 0.0

    def complete(self, system: str, user: str) -> str:
        return self.fn(system, user)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    provider: str = "anthropic"  # "anthropic" | "openai" | "openrouter" | "callable"
    model: str = "claude-sonnet-4-5"
    temperature: Optional[float] = 0.2     # None → omit (gpt-5/o-series)
    max_tokens: int = 2000
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    cache_dir: Optional[str] = ".llm_cache"
    max_schema_retries: int = 2            # re-prompts after validation errors
    max_transport_retries: int = 5         # rate limits / transient failures
    transport_backoff_s: float = 2.0

    @classmethod
    def from_file(cls, path: str) -> "LLMConfig":
        with open(path) as f:
            raw = json.load(f)
        return cls(**raw)

    def make_provider(self):
        kwargs = dict(model=self.model, temperature=self.temperature,
                      max_tokens=self.max_tokens, base_url=self.base_url)
        if self.provider == "anthropic":
            if self.api_key_env:
                kwargs["api_key_env"] = self.api_key_env
            return AnthropicProvider(**kwargs)
        if self.provider == "openai":
            if self.api_key_env:
                kwargs["api_key_env"] = self.api_key_env
            return OpenAICompatProvider(**kwargs)
        if self.provider == "openrouter":
            # OpenRouter is OpenAI-compatible; model ids look like
            # "anthropic/claude-sonnet-4.5" or "openai/gpt-5-mini"
            kwargs["base_url"] = self.base_url or OPENROUTER_BASE_URL
            kwargs["api_key_env"] = self.api_key_env or "OPENROUTER_API_KEY"
            return OpenAICompatProvider(**kwargs)
        raise ValueError(f"Unknown provider '{self.provider}'")


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


class PromptCache:
    """File-per-entry JSON cache keyed by (model, temperature, system, user).

    Writes are atomic (tmp + rename) so concurrent trial workers can share
    one cache directory."""

    def __init__(self, directory: str):
        self.dir = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, key + ".json")

    @staticmethod
    def key(model: str, temperature: Optional[float], system: str, user: str) -> str:
        h = hashlib.sha256()
        temp = "default" if temperature is None else f"{temperature:.4f}"
        h.update(f"{model}|{temp}|".encode())
        h.update(system.encode())
        h.update(b"|")
        h.update(user.encode())
        return h.hexdigest()

    def get(self, key: str) -> Optional[str]:
        try:
            with open(self._path(key)) as f:
                return json.load(f)["response"]
        except (OSError, ValueError, KeyError):
            return None

    def put(self, key: str, response: str) -> None:
        tmp = self._path(key) + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"response": response}, f)
        os.replace(tmp, self._path(key))


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> str:
    """Pull the first JSON object out of an LLM response (handles markdown
    fences and leading/trailing prose)."""
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in response")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError("unbalanced JSON object in response")


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class LLMAgent(BaseAgent):
    def __init__(self, config: LLMConfig, provider=None, name: Optional[str] = None):
        self.config = config
        self.provider = provider or config.make_provider()
        self.name = name or f"llm:{config.model}"
        self.cache = PromptCache(config.cache_dir) if config.cache_dir else None
        self.llm_failures = 0          # schema fallbacks
        self.transport_failures = 0

    def reset(self) -> None:
        pass  # failure counters intentionally persist across a run

    def preflight(self) -> None:
        """Cheap sanity check before a run burns seeds: a missing API key
        would otherwise degrade every year to the hold-prior fallback and
        produce a 'successful' run of garbage."""
        env = getattr(self.provider, "api_key_env", None)
        if env and not os.environ.get(env):
            raise ProviderError(
                f"environment variable {env} is not set — export it (or use "
                f"a config with the right api_key_env) before running "
                f"agent '{self.name}'")

    # -- prompt assembly ----------------------------------------------------

    def build_prompt(self, obs: Observation, error_feedback: str = "") -> str:
        parts = [
            obs.render_text(),
            "",
            "MACHINE-READABLE STATE (same information as above, as JSON):",
            json.dumps(obs.to_json_dict(), sort_keys=True),
            "",
            "OUTPUT JSON SCHEMA (respond with exactly one object of this shape):",
            json.dumps(decision_json_schema(), sort_keys=True),
        ]
        if error_feedback:
            parts += ["", "YOUR PREVIOUS RESPONSE FAILED VALIDATION:",
                      error_feedback,
                      "Fix the problem and respond again with ONLY the JSON object."]
        return "\n".join(parts)

    # -- transport ------------------------------------------------------------

    def _complete(self, system: str, user: str) -> str:
        if self.cache is not None:
            key = PromptCache.key(self.config.model, self.config.temperature,
                                  system, user)
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        delay = self.config.transport_backoff_s
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_transport_retries):
            try:
                text = self.provider.complete(system, user)
                if self.cache is not None:
                    self.cache.put(key, text)
                return text
            except Exception as e:  # rate limits, transient network, 5xx
                last_exc = e
                self.transport_failures += 1
                if attempt < self.config.max_transport_retries - 1:
                    time.sleep(delay)
                    delay *= 2
        raise ProviderError(f"provider failed after "
                            f"{self.config.max_transport_retries} attempts: "
                            f"{last_exc}") from last_exc

    # -- decision -----------------------------------------------------------------

    def decide(self, obs: Observation) -> Decision:
        error_feedback = ""
        for _attempt in range(1 + self.config.max_schema_retries):
            prompt = self.build_prompt(obs, error_feedback)
            try:
                raw = self._complete(self.config.system_prompt, prompt)
            except ProviderError:
                break  # transport dead → fallback
            try:
                payload = extract_json_object(raw)
                return Decision.model_validate_json(payload)
            except (ValueError, ValidationError) as e:
                error_feedback = str(e)[:2000]
        self.llm_failures += 1
        state = HouseholdState.from_dict(obs.state)
        return Decision.hold_prior(state)


# ---------------------------------------------------------------------------
# Mock LLM (deterministic, plausible-but-imperfect; used by the example
# comparison report and the end-to-end smoke test)
# ---------------------------------------------------------------------------


class MockLLMAgent(LLMAgent):
    """A canned 'LLM' with a believable mid-quality policy. Its decisions are
    emitted as JSON *text* and run through the real extraction/validation
    pipeline, so the whole LLM plumbing is exercised without a network.

    Deliberate imperfections (an eval needs a distinguishable mid player):
    flat 80/20 allocation until 55 (no glide, no asset location), skips the
    backdoor when phased out, claims SS at 65, never Roth-converts, never
    harvests losses, and drains pre-tax accounts first in retirement.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        cfg = LLMConfig(provider="callable", model="mock-llm", temperature=0.0,
                        cache_dir=cache_dir)
        super().__init__(cfg, provider=CallableProvider(self._policy_text,
                                                        model="mock-llm"),
                         name="mock-llm")
        self._obs: Optional[Observation] = None

    def decide(self, obs: Observation) -> Decision:
        self._obs = obs  # the "endpoint" reads the observation, not the prompt
        return super().decide(obs)

    # the fake endpoint ------------------------------------------------------

    def _policy_text(self, system: str, user: str) -> str:
        obs = self._obs
        assert obs is not None
        st = HouseholdState.from_dict(obs.state)
        from ..core.accounts import ContributionLimits
        limits = ContributionLimits.for_year(st.inflation_index, st.filing_status)

        avg_age = sum(a.age for a in st.adults) / len(st.adults)
        equity = 0.80 if avg_age < 55 else 0.60
        alloc = {"stocks_us": round(equity * 0.7, 4),
                 "stocks_intl": round(equity * 0.3, 4),
                 "bonds": round(1 - equity, 4), "cash": 0.0}
        allocations = {aid: dict(alloc) for aid in st.accounts if aid != "cash"}

        d: dict = {
            "rationale": "Save steadily, simple 80/20 growth tilt, fund 529s, "
                         "claim SS at 65, spend to plan in retirement.",
            "annual_spending_discretionary": 0.0,
            "contributions": {}, "withdrawals": {},
            "allocations": allocations,
            "retire_now": {"adult1": False, "adult2": False},
            "claim_social_security": {"adult1": False, "adult2": False},
            "tax_loss_harvest": False,
            "roth_conversion_amount": 0.0,
        }
        contribs = d["contributions"]

        def working(a) -> bool:  # engine retires at retirement_age year-start
            return a is not None and a.working and a.age < a.retirement_age

        gross = sum(a.gross_salary for a in st.adults if working(a))
        mortgage = (min(12 * st.mortgage.monthly_payment,
                        st.mortgage.balance * 1.05)
                    if st.mortgage and st.mortgage.active else 0.0)
        all_retired = not any(working(a) for a in st.adults)

        if not all_retired:
            est_magi = gross
            for adult, key in ((st.adult("a1"), "401k_a1"), (st.adult("a2"), "401k_a2")):
                if working(adult):
                    amt = min(0.10 * adult.gross_salary,
                              limits.max_401k_employee(adult.age))
                    contribs[key] = round(amt, 2)
                    est_magi -= amt
            if est_magi < limits.roth_magi_phaseout_lo - 10_000:
                for adult, key in ((st.adult("a1"), "roth_ira_a1"),
                                   (st.adult("a2"), "roth_ira_a2")):
                    if adult:
                        contribs[key] = round(min(5_000 * st.inflation_index,
                                                  limits.max_ira(adult.age)), 2)
            for child in st.children:
                if child.age < child.college_start_age:
                    contribs[f"529_{child.person_id}"] = round(
                        min(6_000 * st.inflation_index,
                            limits.limit_529_per_child), 2)
            cash = st.accounts["cash"].balance
            buffer = st.essential_expenses + mortgage
            if cash > buffer:
                contribs["taxable"] = round(cash - buffer, 2)
            disc = 0.66 * gross - st.essential_expenses - mortgage
            d["annual_spending_discretionary"] = round(
                min(max(disc, st.discretionary_floor), st.discretionary_ceiling), 2)
        else:
            target = st.goals.retirement_spending_real * st.inflation_index
            ss = sum(a.ss_annual_benefit for a in st.adults if a.ss_claimed)
            disc = min(max(target - st.essential_expenses, st.discretionary_floor),
                       st.discretionary_ceiling)
            d["annual_spending_discretionary"] = round(disc, 2)
            need = max(0.0, (st.essential_expenses + mortgage + disc) * 1.10 - ss
                       - st.accounts["cash"].balance)
            trad_bal = sum(st.accounts[a].balance for a in
                           ("trad_401k_a1", "trad_401k_a2") if a in st.accounts)
            w401 = min(need, trad_bal)
            d["withdrawals"]["trad_401k"] = round(w401, 2)
            need -= w401
            if need > 0 and "taxable" in st.accounts:
                take = min(need, st.accounts["taxable"].balance)
                d["withdrawals"]["taxable"] = round(take, 2)
                need -= take
            if need > 0:
                roth_bal = sum(st.accounts[a].balance for a in
                               ("roth_ira_a1", "roth_ira_a2") if a in st.accounts)
                d["withdrawals"]["roth"] = round(min(need, roth_bal), 2)

        for adult, key in ((st.adult("a1"), "adult1"), (st.adult("a2"), "adult2")):
            if adult and not adult.ss_claimed and adult.age >= 65:
                d["claim_social_security"][key] = True

        return json.dumps(d)
