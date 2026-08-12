"""Life-event mechanics: retirement, Social Security claiming, college windows.

Deaths are deterministic (both adults live to exactly 95 — see README), so
there is no mortality machinery; the horizon simply runs until the younger
adult reaches 95.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .accounts import (SS_LATEST_CLAIM_AGE, ss_claiming_factor)
from .state import Child, HouseholdState, Person


@dataclass
class LifeEventLog:
    """Human-relevant transitions that occurred this sim year."""

    retirements: list[str] = field(default_factory=list)       # person_ids
    ss_claims: list[dict] = field(default_factory=list)        # {person_id, age, factor, forced}
    college_started: list[str] = field(default_factory=list)   # child ids
    college_finished: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "retirements": self.retirements,
            "ss_claims": self.ss_claims,
            "college_started": self.college_started,
            "college_finished": self.college_finished,
        }


def apply_retirement_transitions(state: HouseholdState,
                                 retire_now_ids: set[str],
                                 events: LifeEventLog) -> None:
    """Retire adults at the start of the year, either because their planned
    retirement age arrived or because the agent said ``retire_now``.

    A retirement takes effect immediately: no salary is earned in the year
    it happens."""
    for adult in state.adults:
        if adult.retired:
            continue
        if adult.age >= adult.retirement_age or adult.person_id in retire_now_ids:
            adult.retired = True
            adult.retirement_age = min(adult.retirement_age, adult.age)
            events.retirements.append(adult.person_id)


def apply_ss_claims(state: HouseholdState, claim_ids: set[str],
                    events: LifeEventLog) -> None:
    """Claim Social Security for the given adults (validated upstream:
    age >= 62, not already claimed). Additionally force-claims anyone who
    reaches 70 unclaimed — delaying past 70 has zero benefit, so the engine
    never lets an agent (or its absence, in drawdown mode) forget entirely.

    The benefit is set from the COLA-indexed PIA and the claiming-age factor,
    and is paid for the full claim year (annual granularity)."""
    for adult in state.adults:
        if adult.ss_claimed:
            continue
        forced = adult.age >= SS_LATEST_CLAIM_AGE
        if adult.person_id in claim_ids or forced:
            claim_age = min(adult.age, SS_LATEST_CLAIM_AGE)
            factor = ss_claiming_factor(claim_age)
            adult.ss_claimed = True
            adult.ss_claim_age = claim_age
            adult.ss_annual_benefit = adult.ss_pia_annual * factor
            events.ss_claims.append({
                "person_id": adult.person_id, "age": adult.age,
                "factor": round(factor, 4), "forced": forced,
            })


def college_transitions(children: list[Child], events: LifeEventLog) -> None:
    """Record college start/finish transitions for this year's ages."""
    for c in children:
        if c.age == c.college_start_age:
            events.college_started.append(c.person_id)
        if c.age == c.college_end_age:
            events.college_finished.append(c.person_id)


def college_costs_due(state: HouseholdState) -> dict[str, float]:
    """Nominal college cost billed this year, per child currently enrolled."""
    return {c.person_id: c.college_annual_cost
            for c in state.children if c.in_college()}


def years_until_both_reach(state: HouseholdState, target_age: int = 95) -> int:
    """Sim years remaining until every adult has reached ``target_age``."""
    if not state.adults:
        return 0
    return max(0, max(target_age - a.age for a in state.adults))
