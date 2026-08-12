"""Scenario library.

A ``Scenario`` is a frozen description of a household at year 0 plus the
evaluation horizon. ``build_initial_state()`` constructs a fresh
``HouseholdState`` (never shared between trials — trials mutate state).

Four scenarios ship with the harness:

* ``meridian``      — the default two-earner couple with two kids (spec §8).
* ``summit``        — high-earner couple approaching early retirement; the
                      Roth-conversion / tax-planning stress test (includes a
                      pre-tax trad IRA that booby-traps naive backdoor Roths
                      via the pro-rata rule).
* ``foothill``      — late-start savers at 50 with modest balances.
* ``harbor``        — single parent (Head of Household) with tight cash flow;
                      the shortfall-management stress test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..core.state import (Account, AccountKind, Allocation, Child, GoalSet,
                          HouseholdState, Mortgage, Person)

DEFAULT_START_YEAR = 2025


def _alloc(us: float = 0.0, intl: float = 0.0, bonds: float = 0.0,
           cash: float = 0.0) -> Allocation:
    return Allocation(stocks_us=us, stocks_intl=intl, bonds=bonds, cash=cash)


@dataclass(frozen=True)
class AccountSpec:
    account_id: str
    kind: AccountKind
    owner: Optional[str]
    balance: float
    basis: float = 0.0
    alloc: Allocation = field(default_factory=lambda: _alloc(cash=1.0))


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    filing_status: str
    adults: tuple
    children: tuple
    accounts: tuple
    mortgage: Optional[Mortgage]
    essential_expenses: float          # excl. mortgage P&I, today's dollars
    discretionary_floor: float
    discretionary_ceiling: float
    hsa_enabled: bool
    goals: GoalSet
    active_years: int = 30
    start_year: int = DEFAULT_START_YEAR

    def build_initial_state(self) -> HouseholdState:
        st = HouseholdState(
            year_index=0,
            start_year=self.start_year,
            filing_status=self.filing_status,
            adults=[Person(**a) for a in self.adults],
            children=[Child(**c) for c in self.children],
            accounts={s.account_id: Account(
                account_id=s.account_id, kind=s.kind, owner=s.owner,
                balance=s.balance, basis=s.basis,
                alloc=Allocation(**s.alloc.as_dict()))
                for s in self.accounts},
            mortgage=Mortgage(**self.mortgage.to_dict()) if self.mortgage else None,
            essential_expenses=self.essential_expenses,
            discretionary_floor=self.discretionary_floor,
            discretionary_ceiling=self.discretionary_ceiling,
            hsa_enabled=self.hsa_enabled,
            goals=GoalSet(**self.goals.to_dict()),
        )
        # every household carries the full account skeleton so decisions and
        # cascades never hit a missing account (zero-balance where unused)
        skeleton = [
            ("taxable", AccountKind.TAXABLE, None),
            ("cash", AccountKind.CASH, None),
        ]
        for adult in st.adults:
            pid = adult.person_id
            skeleton += [
                (f"trad_401k_{pid}", AccountKind.TRAD_401K, pid),
                (f"roth_ira_{pid}", AccountKind.ROTH_IRA, pid),
                (f"trad_ira_{pid}", AccountKind.TRAD_IRA, pid),
            ]
        if self.hsa_enabled:
            skeleton.append(("hsa", AccountKind.HSA, "a1"))
        for ch in st.children:
            skeleton.append((f"529_{ch.person_id}", AccountKind.COLLEGE_529,
                             ch.person_id))
        for aid, kind, owner in skeleton:
            if aid not in st.accounts:
                st.accounts[aid] = Account(account_id=aid, kind=kind, owner=owner,
                                           balance=0.0, alloc=_alloc(cash=1.0))
        return st

    def total_years(self) -> int:
        """Sim years until both adults have reached 95."""
        return max(95 - a["age"] for a in self.adults)


# ---------------------------------------------------------------------------
# 1. Meridian family (default; spec §8)
# ---------------------------------------------------------------------------

_GROWTH_60 = _alloc(us=0.60, intl=0.15, bonds=0.20, cash=0.05)
_GROWTH_70 = _alloc(us=0.55, intl=0.20, bonds=0.20, cash=0.05)

MERIDIAN = Scenario(
    name="meridian",
    description=("Two-earner couple (38/36, $145k/$95k) with kids 6 and 3; "
                 "$420k mortgage; college goal 80% of $40k/yr/child sticker; "
                 "retire at 65 targeting $90k/yr real spending."),
    filing_status="mfj",
    adults=(
        dict(person_id="a1", age=38, retirement_age=65, gross_salary=145_000.0,
             salary_growth=0.03, ss_pia_annual=3_200.0 * 12,
             match_rate=0.5, match_cap_fraction=0.06),
        dict(person_id="a2", age=36, retirement_age=65, gross_salary=95_000.0,
             salary_growth=0.03, ss_pia_annual=2_100.0 * 12,
             match_rate=0.5, match_cap_fraction=0.06),
    ),
    children=(
        dict(person_id="child1", age=6, college_start_age=18, college_years=4,
             college_annual_cost=40_000.0),
        dict(person_id="child2", age=3, college_start_age=18, college_years=4,
             college_annual_cost=40_000.0),
    ),
    accounts=(
        AccountSpec("taxable", AccountKind.TAXABLE, None, 60_000.0,
                    basis=45_000.0, alloc=_GROWTH_70),           # basis ratio 0.75
        AccountSpec("trad_401k_a1", AccountKind.TRAD_401K, "a1", 180_000.0,
                    alloc=_GROWTH_70),
        AccountSpec("trad_401k_a2", AccountKind.TRAD_401K, "a2", 90_000.0,
                    alloc=_GROWTH_70),
        AccountSpec("roth_ira_a1", AccountKind.ROTH_IRA, "a1", 40_000.0,
                    basis=30_000.0, alloc=_GROWTH_70),
        AccountSpec("roth_ira_a2", AccountKind.ROTH_IRA, "a2", 25_000.0,
                    basis=20_000.0, alloc=_GROWTH_70),
        AccountSpec("529_child1", AccountKind.COLLEGE_529, "child1", 15_000.0,
                    basis=12_000.0, alloc=_GROWTH_60),
        AccountSpec("529_child2", AccountKind.COLLEGE_529, "child2", 8_000.0,
                    basis=6_500.0, alloc=_GROWTH_60),
        AccountSpec("cash", AccountKind.CASH, None, 35_000.0),
    ),
    # $110k/yr "essential + mortgage": P&I on the stated terms is
    # $2,470/mo = $29,640/yr, leaving $80,360 of non-mortgage essentials.
    # (At $2,470/mo the $420k @ 4.25% balance amortizes in ~22 years; the
    # spec's "~27 years left" and payment are mutually approximate — the
    # engine honors balance/rate/payment and lets the term emerge.)
    mortgage=Mortgage(balance=420_000.0, annual_rate=0.0425,
                      monthly_payment=2_470.0),
    essential_expenses=80_360.0,
    discretionary_floor=8_000.0,
    discretionary_ceiling=80_000.0,
    hsa_enabled=False,
    goals=GoalSet(college_fund_fraction=0.80,
                  retirement_spending_real=90_000.0,
                  bequest_target_real=None),
)


# ---------------------------------------------------------------------------
# 2. Summit — high-earner tax-planning stress test
# ---------------------------------------------------------------------------

SUMMIT = Scenario(
    name="summit",
    description=("High earners (52/50, $320k/$180k) targeting retirement at "
                 "60: a Roth-conversion window from 60 to RMDs at 73, NIIT "
                 "and Additional-Medicare exposure, a pre-tax trad IRA that "
                 "makes careless backdoor Roths pro-rata taxable, and a "
                 "nearly-funded 529 for a 16-year-old."),
    filing_status="mfj",
    adults=(
        dict(person_id="a1", age=52, retirement_age=60, gross_salary=320_000.0,
             salary_growth=0.02, ss_pia_annual=4_000.0 * 12,
             match_rate=0.5, match_cap_fraction=0.06),
        dict(person_id="a2", age=50, retirement_age=60, gross_salary=180_000.0,
             salary_growth=0.02, ss_pia_annual=2_800.0 * 12,
             match_rate=0.5, match_cap_fraction=0.06),
    ),
    children=(
        dict(person_id="child1", age=16, college_start_age=18, college_years=4,
             college_annual_cost=55_000.0),
    ),
    accounts=(
        AccountSpec("taxable", AccountKind.TAXABLE, None, 900_000.0,
                    basis=540_000.0, alloc=_GROWTH_70),
        AccountSpec("trad_401k_a1", AccountKind.TRAD_401K, "a1", 1_400_000.0,
                    alloc=_GROWTH_60),
        AccountSpec("trad_401k_a2", AccountKind.TRAD_401K, "a2", 700_000.0,
                    alloc=_GROWTH_60),
        AccountSpec("trad_ira_a1", AccountKind.TRAD_IRA, "a1", 250_000.0,
                    basis=0.0, alloc=_GROWTH_60),
        AccountSpec("roth_ira_a1", AccountKind.ROTH_IRA, "a1", 150_000.0,
                    basis=90_000.0, alloc=_GROWTH_70),
        AccountSpec("roth_ira_a2", AccountKind.ROTH_IRA, "a2", 100_000.0,
                    basis=60_000.0, alloc=_GROWTH_70),
        AccountSpec("hsa", AccountKind.HSA, "a1", 60_000.0, alloc=_GROWTH_70),
        AccountSpec("529_child1", AccountKind.COLLEGE_529, "child1", 180_000.0,
                    basis=120_000.0, alloc=_alloc(us=0.25, intl=0.05,
                                                  bonds=0.50, cash=0.20)),
        AccountSpec("cash", AccountKind.CASH, None, 120_000.0),
    ),
    mortgage=Mortgage(balance=300_000.0, annual_rate=0.030,
                      monthly_payment=2_800.0),
    essential_expenses=110_000.0,
    discretionary_floor=20_000.0,
    discretionary_ceiling=150_000.0,
    hsa_enabled=True,
    goals=GoalSet(college_fund_fraction=1.00,
                  retirement_spending_real=160_000.0,
                  bequest_target_real=2_000_000.0),
)


# ---------------------------------------------------------------------------
# 3. Foothill — late-start savers at 50
# ---------------------------------------------------------------------------

FOOTHILL = Scenario(
    name="foothill",
    description=("Both 50 with only $100k saved and a 5.5% mortgage: "
                 "catch-up contributions, working-longer trade-offs, and "
                 "Social Security timing dominate."),
    filing_status="mfj",
    adults=(
        dict(person_id="a1", age=50, retirement_age=67, gross_salary=105_000.0,
             salary_growth=0.03, ss_pia_annual=2_600.0 * 12,
             match_rate=0.5, match_cap_fraction=0.06),
        dict(person_id="a2", age=50, retirement_age=67, gross_salary=70_000.0,
             salary_growth=0.03, ss_pia_annual=1_900.0 * 12,
             match_rate=0.5, match_cap_fraction=0.06),
    ),
    children=(),
    accounts=(
        AccountSpec("taxable", AccountKind.TAXABLE, None, 20_000.0,
                    basis=18_000.0, alloc=_GROWTH_60),
        AccountSpec("trad_401k_a1", AccountKind.TRAD_401K, "a1", 60_000.0,
                    alloc=_GROWTH_60),
        AccountSpec("trad_401k_a2", AccountKind.TRAD_401K, "a2", 25_000.0,
                    alloc=_GROWTH_60),
        AccountSpec("hsa", AccountKind.HSA, "a1", 5_000.0, alloc=_GROWTH_60),
        AccountSpec("cash", AccountKind.CASH, None, 15_000.0),
    ),
    mortgage=Mortgage(balance=240_000.0, annual_rate=0.055,
                      monthly_payment=1_900.0),
    essential_expenses=62_000.0,
    discretionary_floor=6_000.0,
    discretionary_ceiling=50_000.0,
    hsa_enabled=True,
    goals=GoalSet(college_fund_fraction=0.0,
                  retirement_spending_real=70_000.0,
                  bequest_target_real=None),
)


# ---------------------------------------------------------------------------
# 4. Harbor — single parent, tight cash flow (Head of Household)
# ---------------------------------------------------------------------------

HARBOR = Scenario(
    name="harbor",
    description=("Single parent (40, $78k, renting) with an 8-year-old: "
                 "every dollar is contested between the emergency fund, the "
                 "529, and retirement — the shortfall-management stress test."),
    filing_status="hoh",
    adults=(
        dict(person_id="a1", age=40, retirement_age=67, gross_salary=78_000.0,
             salary_growth=0.025, ss_pia_annual=2_200.0 * 12,
             match_rate=0.5, match_cap_fraction=0.06),
    ),
    children=(
        dict(person_id="child1", age=8, college_start_age=18, college_years=4,
             college_annual_cost=30_000.0),
    ),
    accounts=(
        AccountSpec("taxable", AccountKind.TAXABLE, None, 8_000.0,
                    basis=7_600.0, alloc=_GROWTH_60),
        AccountSpec("trad_401k_a1", AccountKind.TRAD_401K, "a1", 45_000.0,
                    alloc=_GROWTH_70),
        AccountSpec("roth_ira_a1", AccountKind.ROTH_IRA, "a1", 12_000.0,
                    basis=10_000.0, alloc=_GROWTH_70),
        AccountSpec("529_child1", AccountKind.COLLEGE_529, "child1", 5_000.0,
                    basis=4_500.0, alloc=_GROWTH_60),
        AccountSpec("cash", AccountKind.CASH, None, 10_000.0),
    ),
    mortgage=None,  # renting; rent is inside essential expenses
    essential_expenses=58_000.0,
    discretionary_floor=4_000.0,
    discretionary_ceiling=40_000.0,
    hsa_enabled=False,
    goals=GoalSet(college_fund_fraction=0.60,
                  retirement_spending_real=55_000.0,
                  bequest_target_real=None),
)


SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in (MERIDIAN, SUMMIT, FOOTHILL, HARBOR)
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. "
                       f"Available: {sorted(SCENARIOS)}")
    return SCENARIOS[name]
