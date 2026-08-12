"""Agent interface: Observation in → Decision out.

``Decision`` is the strict JSON schema every agent (LLM or rule-based) must
produce. Field names follow the eval spec exactly (pydantic aliases handle
the leading-digit keys like ``401k_a1``). Unknown keys are rejected —
hallucinated fields should fail validation loudly, not pass silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..core.state import ASSET_CLASSES, HouseholdState


# ---------------------------------------------------------------------------
# Decision schema
# ---------------------------------------------------------------------------


class Contributions(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    k401_a1: float = Field(0.0, alias="401k_a1", ge=0)
    k401_a2: float = Field(0.0, alias="401k_a2", ge=0)
    roth_ira_a1: float = Field(0.0, ge=0)
    roth_ira_a2: float = Field(0.0, ge=0)
    backdoor_roth_a1: bool = False
    backdoor_roth_a2: bool = False
    hsa: float = Field(0.0, ge=0)
    c529_child1: float = Field(0.0, alias="529_child1", ge=0)
    c529_child2: float = Field(0.0, alias="529_child2", ge=0)
    taxable: float = Field(0.0, ge=0)
    extra_mortgage_principal: float = Field(0.0, ge=0)


class Withdrawals(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    taxable: float = Field(0.0, ge=0)
    trad_401k: float = Field(0.0, ge=0)
    trad_ira: float = Field(0.0, ge=0)  # optional extension of the spec schema
    roth: float = Field(0.0, ge=0)
    hsa: float = Field(0.0, ge=0)


class AdultFlags(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    adult1: bool = False
    adult2: bool = False

    def for_person(self, person_id: str) -> bool:
        return self.adult1 if person_id == "a1" else self.adult2


class Decision(BaseModel):
    """One year of financial decisions. Strict schema (extra keys rejected).

    ``allocations`` maps account_id -> {asset_class: weight}; accounts left
    out keep their prior target allocation. Dollar fields are nominal
    (current-year) dollars.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    rationale: str = ""
    annual_spending_discretionary: float = Field(0.0, ge=0)
    contributions: Contributions = Field(default_factory=Contributions)
    roth_conversion_amount: float = Field(0.0, ge=0)
    withdrawals: Withdrawals = Field(default_factory=Withdrawals)
    allocations: dict[str, dict[str, float]] = Field(default_factory=dict)
    retire_now: AdultFlags = Field(default_factory=AdultFlags)
    claim_social_security: AdultFlags = Field(default_factory=AdultFlags)
    tax_loss_harvest: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> "Decision":
        return cls.model_validate(d)

    @classmethod
    def hold_prior(cls, state: HouseholdState) -> "Decision":
        """Fallback decision: repeat last year's, with one-shot flags cleared.

        Used when an LLM fails schema validation after retries. If there is
        no prior decision, a conservative all-zeros decision (spending at the
        floor, hold allocations) is returned.
        """
        if not state.prior_decision:
            return cls(rationale="fallback: no prior decision; hold everything",
                       annual_spending_discretionary=state.discretionary_floor)
        d = json.loads(json.dumps(state.prior_decision))  # deep copy
        d["retire_now"] = {"adult1": False, "adult2": False}
        d["claim_social_security"] = {"adult1": False, "adult2": False}
        d["rationale"] = "fallback: held prior year's decision (schema failure)"
        d["allocations"] = {}  # keep prior targets implicitly
        return cls.model_validate(d)


def decision_json_schema() -> dict[str, Any]:
    """JSON schema (by alias) given to LLM agents in the prompt."""
    return Decision.model_json_schema(by_alias=True)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """Everything an agent may see for one decision year.

    Strictly backward-looking: realized returns up to last year, current
    balances, known schedules (salaries, college timing, benefit formulas).
    No forward-looking market information, ever.
    """

    year_index: int
    calendar_year: int
    state: dict[str, Any]                      # full HouseholdState.to_dict()
    limits: dict[str, float]                   # this year's contribution limits
    rules_text: str                            # plain-language statement of rules
    upcoming: list[str]                        # known obligations, human-readable
    goals_text: str
    recent_returns: list[dict[str, float]]     # last <=5 years, realized
    prior_decision: Optional[dict[str, Any]]
    prior_rationale: str
    estimated_magi: float                      # provisional MAGI (phase-out info)
    rmd_due: dict[str, float] = dc_field(default_factory=dict)
    scenario_name: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "year_index": self.year_index,
            "calendar_year": self.calendar_year,
            "scenario": self.scenario_name,
            # history is rendered as its own JSON-per-line prompt section;
            # repeating it inside the state dump would double its token cost
            "household_state": {k: v for k, v in self.state.items()
                                if k != "history"},
            "contribution_limits": self.limits,
            "estimated_magi": self.estimated_magi,
            "rmd_due": self.rmd_due,
            "recent_realized_returns": self.recent_returns,
            "upcoming_obligations": self.upcoming,
            "goals": self.goals_text,
            "prior_decision": self.prior_decision,
            "prior_rationale": self.prior_rationale,
        }

    # -- plain-language rendering (the LLM-facing summary) -----------------

    def render_text(self) -> str:
        s = self.state
        lines: list[str] = []
        lines.append(f"=== Year {self.year_index} (calendar {self.calendar_year}) — "
                     f"scenario '{self.scenario_name}' ===")
        lines.append("")
        lines.append("PEOPLE")
        for a in s["adults"]:
            status = "retired" if a["retired"] else \
                f"working, salary ${a['gross_salary']:,.0f}/yr (growth {a['salary_growth']:.1%})"
            ss = "claimed" if a["ss_claimed"] else \
                f"not claimed (PIA ${a['ss_pia_annual']:,.0f}/yr at FRA 67)"
            if a["ss_claimed"]:
                ss += f", benefit ${a['ss_annual_benefit']:,.0f}/yr"
            lines.append(f"  {a['person_id']}: age {a['age']}, planned retirement "
                         f"{a['retirement_age']}, {status}; Social Security: {ss}")
        for c in s["children"]:
            lines.append(f"  {c['person_id']}: age {c['age']}, college at "
                         f"{c['college_start_age']} for {c['college_years']} years, "
                         f"current sticker ${c['college_annual_cost']:,.0f}/yr")
        lines.append("")
        lines.append("ACCOUNTS (balance | target allocation)")
        for aid, acct in sorted(s["accounts"].items()):
            al = acct["alloc"]
            alloc_txt = ", ".join(f"{k} {al[k]:.0%}" for k in ASSET_CLASSES if al[k] > 0.001)
            extra = ""
            if acct["kind"] == "taxable" and acct["balance"] > 0:
                extra = f" | basis ratio {min(1.0, acct['basis'] / acct['balance']):.2f}"
            lines.append(f"  {aid}: ${acct['balance']:,.0f} | {alloc_txt or 'all cash'}{extra}")
        if s.get("mortgage"):
            mg = s["mortgage"]
            lines.append(f"  mortgage: balance ${mg['balance']:,.0f} at "
                         f"{mg['annual_rate']:.2%}, P&I ${mg['monthly_payment']:,.0f}/mo")
        lines.append("")
        lines.append("CASH FLOW (current nominal $)")
        lines.append(f"  essential expenses (excl. mortgage): ${s['essential_expenses']:,.0f}/yr")
        lines.append(f"  discretionary spending range: ${s['discretionary_floor']:,.0f} – "
                     f"${s['discretionary_ceiling']:,.0f}/yr")
        lines.append(f"  estimated household MAGI this year: ${self.estimated_magi:,.0f}")
        if self.rmd_due:
            for pid, amt in self.rmd_due.items():
                lines.append(f"  REQUIRED minimum distribution for {pid}: ${amt:,.0f} "
                             f"(auto-enforced if you withdraw less)")
        lines.append("")
        lines.append("GOALS (priority order)")
        lines.append(self.goals_text)
        lines.append("")
        if self.upcoming:
            lines.append("UPCOMING OBLIGATIONS")
            for u in self.upcoming:
                lines.append(f"  - {u}")
            lines.append("")
        if self.recent_returns:
            lines.append("REALIZED RETURNS, LAST YEARS (most recent last; nominal)")
            hdr = "  year  us_stocks  intl  bonds  cash  inflation"
            lines.append(hdr)
            n = len(self.recent_returns)
            for i, r in enumerate(self.recent_returns):
                lines.append(f"  t-{n - i:<3} {r['stocks_us']:>8.1%} {r['stocks_intl']:>6.1%} "
                             f"{r['bonds']:>6.1%} {r['cash']:>5.1%} {r['inflation']:>8.1%}")
            lines.append("")
        lines.append("THIS YEAR'S CONTRIBUTION LIMITS (already inflation-indexed)")
        for k, v in self.limits.items():
            lines.append(f"  {k}: ${v:,.0f}")
        lines.append("")
        history = s.get("history") or []
        if history:
            lines.append("YOUR FULL HISTORY (every prior year: decision digest, "
                         "your rationale verbatim, and the realized outcome — "
                         "one JSON object per year)")
            for rec in history:
                lines.append("  " + json.dumps(rec, sort_keys=True))
            lines.append("")
        if self.prior_decision is not None:
            lines.append("YOUR PRIOR-YEAR DECISION (for continuity; avoid thrashing)")
            lines.append("  " + json.dumps(self.prior_decision, sort_keys=True))
            lines.append(f"  prior rationale: {self.prior_rationale}")
            lines.append("")
        lines.append("RULES")
        lines.append(self.rules_text)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------


class BaseAgent:
    """state in → Decision out. Subclasses must be deterministic given the
    observation (any LLM nondeterminism is confined to the API call)."""

    name: str = "base"

    def decide(self, obs: Observation) -> Decision:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the start of each trial (seed). Stateless by default."""


class ScriptedAgent(BaseAgent):
    """Replays a fixed list of decisions (used by determinism tests)."""

    name = "scripted"

    def __init__(self, decisions: list[Decision], pad_with_last: bool = True):
        self._decisions = decisions
        self._pad = pad_with_last

    def decide(self, obs: Observation) -> Decision:
        i = obs.year_index
        if i < len(self._decisions):
            return self._decisions[i]
        if self._pad and self._decisions:
            return Decision.model_validate(self._decisions[-1].model_dump(by_alias=True))
        return Decision()
