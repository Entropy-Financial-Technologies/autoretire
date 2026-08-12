"""Household state model.

Everything the simulator needs to advance one year lives here, and all of it
serializes to plain JSON (versioned) so trial logs are self-describing and
replayable.

Conventions used throughout the engine
--------------------------------------
* The simulation runs in **nominal dollars**. Realized CPI (from the seeded
  return generator) indexes expenses, tax brackets, and contribution limits.
  Real (today's-dollar) figures are derived by deflating with
  ``inflation_index``.
* ``Person.age`` / ``Child.age`` are the age at the **start** of the sim year.
  Ages advance by one at the end of each simulated year.
* Account ids are stable strings that match the decision schema:
  ``taxable``, ``trad_401k_a1``, ``trad_401k_a2``, ``roth_ira_a1``,
  ``roth_ira_a2``, ``trad_ira_a1``, ``trad_ira_a2``, ``hsa``,
  ``529_child1``, ``529_child2``, ``cash``.
* ``Account.basis`` means different things per kind (documented on the class);
  the tax engine relies on these meanings.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

STATE_SCHEMA_VERSION = "1.0"

ASSET_CLASSES = ("stocks_us", "stocks_intl", "bonds", "cash")


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


@dataclass
class Allocation:
    """Target asset-class weights for one account. Weights sum to 1.

    Accounts are rebalanced to their target allocation once per year when the
    agent's decision is applied (internal rebalancing is modeled as tax-free —
    a documented simplification; only withdrawals realize gains in taxable).
    """

    stocks_us: float = 0.0
    stocks_intl: float = 0.0
    bonds: float = 0.0
    cash: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in ASSET_CLASSES}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "Allocation":
        return cls(**{k: float(d.get(k, 0.0)) for k in ASSET_CLASSES})

    def weighted_return(self, returns: dict[str, float]) -> float:
        """Portfolio return for this allocation given per-asset returns."""
        return sum(getattr(self, k) * returns[k] for k in ASSET_CLASSES)

    def equity_fraction(self) -> float:
        return self.stocks_us + self.stocks_intl

    def total(self) -> float:
        return sum(getattr(self, k) for k in ASSET_CLASSES)

    def normalized(self) -> "Allocation":
        t = self.total()
        if t <= 0:
            return Allocation(0.0, 0.0, 0.0, 1.0)
        return Allocation(*(getattr(self, k) / t for k in ASSET_CLASSES))


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


class AccountKind(str, Enum):
    TAXABLE = "taxable"
    TRAD_401K = "trad_401k"
    ROTH_IRA = "roth_ira"
    TRAD_IRA = "trad_ira"
    HSA = "hsa"
    COLLEGE_529 = "529"
    CASH = "cash"


#: Kinds whose withdrawals are taxed as ordinary income (pre-tax money).
PRETAX_KINDS = (AccountKind.TRAD_401K, AccountKind.TRAD_IRA)


@dataclass
class Account:
    """A single account.

    ``basis`` semantics by kind:

    * ``TAXABLE``  – total cost basis in dollars (single blended basis; the
      basis *ratio* is ``basis / balance``). Contributions and taxed
      distributions (dividends/interest, which are modeled as reinvested)
      increase basis; withdrawals remove basis pro-rata.
    * ``ROTH_IRA`` – contribution + conversion basis. Withdrawals consume
      basis first (tax- and penalty-free); the remainder is earnings.
      (5-year clocks are intentionally not modeled — see README.)
    * ``TRAD_IRA`` – after-tax (nondeductible) basis, nonzero only via
      backdoor-Roth pro-rata leftovers. Withdrawals/conversions recover it
      pro-rata.
    * ``COLLEGE_529`` – contribution basis, used to split earnings on
      non-qualified withdrawals.
    * ``TRAD_401K``, ``HSA``, ``CASH`` – unused (0).
    """

    account_id: str
    kind: AccountKind
    owner: Optional[str]  # "a1"/"a2" for adults, "child1"/"child2" for 529s
    balance: float = 0.0
    basis: float = 0.0
    alloc: Allocation = field(default_factory=Allocation)

    def basis_ratio(self) -> float:
        if self.balance <= 0:
            return 1.0
        return min(1.0, max(0.0, self.basis / self.balance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "kind": self.kind.value,
            "owner": self.owner,
            "balance": self.balance,
            "basis": self.basis,
            "alloc": self.alloc.as_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Account":
        return cls(
            account_id=d["account_id"],
            kind=AccountKind(d["kind"]),
            owner=d.get("owner"),
            balance=float(d["balance"]),
            basis=float(d.get("basis", 0.0)),
            alloc=Allocation.from_dict(d.get("alloc", {"cash": 1.0})),
        )


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------


@dataclass
class Person:
    """An adult household member.

    ``ss_pia_annual`` is the annual Social Security benefit this person would
    receive claiming exactly at full retirement age (FRA, assumed 67), in
    *current nominal dollars* — it receives COLA (realized CPI) every sim year
    until claimed. On claiming, ``ss_annual_benefit`` is set to
    ``pia * claiming_factor(claim_age)`` and receives COLA thereafter.
    """

    person_id: str  # "a1" | "a2"
    age: int
    retirement_age: int
    gross_salary: float  # current nominal annual salary (0 if never worked)
    salary_growth: float  # nominal annual growth rate while working
    ss_pia_annual: float  # annual PIA at FRA, current nominal dollars
    match_rate: float = 0.5  # employer match: rate on matched contributions
    match_cap_fraction: float = 0.06  # ...up to this fraction of salary
    retired: bool = False
    ss_claimed: bool = False
    ss_claim_age: Optional[int] = None
    ss_annual_benefit: float = 0.0  # nominal, set at claim, COLA'd after

    @property
    def working(self) -> bool:
        return not self.retired and self.gross_salary > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Person":
        return cls(**d)


@dataclass
class Child:
    person_id: str  # "child1" | "child2"
    age: int
    college_start_age: int = 18
    college_years: int = 4
    college_annual_cost: float = 0.0  # current nominal sticker per year

    @property
    def college_end_age(self) -> int:
        return self.college_start_age + self.college_years

    def in_college(self) -> bool:
        return self.college_start_age <= self.age < self.college_end_age

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Child":
        return cls(**d)


# ---------------------------------------------------------------------------
# Cash-flow objects
# ---------------------------------------------------------------------------


@dataclass
class Mortgage:
    balance: float
    annual_rate: float
    monthly_payment: float  # P&I only

    @property
    def active(self) -> bool:
        return self.balance > 0.005

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Mortgage":
        return cls(**d)


@dataclass
class GoalSet:
    """Household goals, in priority order (1 = highest).

    1. Essential spending funded every year until both adults reach age 95.
    2. Fund ``college_fund_fraction`` of each child's college cost.
    3. Retirement lifestyle: ``retirement_spending_real`` per year
       (today's dollars, total consumption incl. essential).
    4. Optional bequest target (today's dollars).
    """

    college_fund_fraction: float = 0.8
    retirement_spending_real: float = 0.0
    bequest_target_real: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoalSet":
        return cls(**d)


# ---------------------------------------------------------------------------
# Household state
# ---------------------------------------------------------------------------


@dataclass
class HouseholdState:
    """Complete simulator state at the start of a sim year."""

    # --- time ---
    year_index: int  # 0-based sim year
    start_year: int  # calendar year of year_index 0 (e.g. 2025)

    # --- people ---
    filing_status: str  # "mfj" | "hoh"
    adults: list[Person] = field(default_factory=list)
    children: list[Child] = field(default_factory=list)

    # --- money ---
    accounts: dict[str, Account] = field(default_factory=dict)
    mortgage: Optional[Mortgage] = None

    # --- recurring cash flow (current nominal dollars per year) ---
    essential_expenses: float = 0.0  # excludes mortgage P&I (modeled separately)
    discretionary_floor: float = 0.0
    discretionary_ceiling: float = 0.0
    hsa_enabled: bool = False

    goals: GoalSet = field(default_factory=GoalSet)

    # --- indexation & tax carry-state ---
    inflation_index: float = 1.0  # cumulative CPI factor since start_year
    capital_loss_carryforward: float = 0.0

    # --- agent memory / history ---
    prior_decision: Optional[dict[str, Any]] = None
    prior_rationale: str = ""
    returns_history: list[dict[str, float]] = field(default_factory=list)

    version: str = STATE_SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def calendar_year(self) -> int:
        return self.start_year + self.year_index

    def adult(self, person_id: str) -> Optional[Person]:
        for a in self.adults:
            if a.person_id == person_id:
                return a
        return None

    def child(self, person_id: str) -> Optional[Child]:
        for c in self.children:
            if c.person_id == person_id:
                return c
        return None

    def account(self, account_id: str) -> Optional[Account]:
        return self.accounts.get(account_id)

    def adult_trad_accounts(self, person_id: str) -> list[Account]:
        """This adult's pre-tax accounts (401k + trad IRA), in RMD order."""
        out = []
        for aid in (f"trad_401k_{person_id}", f"trad_ira_{person_id}"):
            acct = self.accounts.get(aid)
            if acct is not None:
                out.append(acct)
        return out

    def total_investable(self) -> float:
        return sum(a.balance for a in self.accounts.values())

    def net_worth(self) -> float:
        """Nominal net worth: all accounts minus mortgage debt.

        The home's value is not modeled; subtracting mortgage debt while
        crediting extra-principal payments keeps prepayment wealth-neutral
        (it converts cash into reduced debt and saves future interest).
        """
        nw = self.total_investable()
        if self.mortgage is not None:
            nw -= self.mortgage.balance
        return nw

    def real(self, nominal: float) -> float:
        """Deflate a current-year nominal amount to start-year dollars."""
        return nominal / self.inflation_index

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "year_index": self.year_index,
            "start_year": self.start_year,
            "filing_status": self.filing_status,
            "adults": [a.to_dict() for a in self.adults],
            "children": [c.to_dict() for c in self.children],
            "accounts": {k: v.to_dict() for k, v in self.accounts.items()},
            "mortgage": self.mortgage.to_dict() if self.mortgage else None,
            "essential_expenses": self.essential_expenses,
            "discretionary_floor": self.discretionary_floor,
            "discretionary_ceiling": self.discretionary_ceiling,
            "hsa_enabled": self.hsa_enabled,
            "goals": self.goals.to_dict(),
            "inflation_index": self.inflation_index,
            "capital_loss_carryforward": self.capital_loss_carryforward,
            "prior_decision": self.prior_decision,
            "prior_rationale": self.prior_rationale,
            "returns_history": self.returns_history,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HouseholdState":
        if d.get("version", STATE_SCHEMA_VERSION) != STATE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported state schema version: {d.get('version')}")
        return cls(
            year_index=d["year_index"],
            start_year=d["start_year"],
            filing_status=d["filing_status"],
            adults=[Person.from_dict(x) for x in d["adults"]],
            children=[Child.from_dict(x) for x in d["children"]],
            accounts={k: Account.from_dict(v) for k, v in d["accounts"].items()},
            mortgage=Mortgage.from_dict(d["mortgage"]) if d.get("mortgage") else None,
            essential_expenses=d["essential_expenses"],
            discretionary_floor=d["discretionary_floor"],
            discretionary_ceiling=d["discretionary_ceiling"],
            hsa_enabled=d["hsa_enabled"],
            goals=GoalSet.from_dict(d["goals"]),
            inflation_index=d["inflation_index"],
            capital_loss_carryforward=d.get("capital_loss_carryforward", 0.0),
            prior_decision=d.get("prior_decision"),
            prior_rationale=d.get("prior_rationale", ""),
            returns_history=d.get("returns_history", []),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def clone(self) -> "HouseholdState":
        return copy.deepcopy(self)
