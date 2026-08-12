"""Baseline sanity (spec §10): the rule-based expert must beat the naive
baseline in CE wealth on ≥90% of seeds — if not, the engine has a bug.
Also: baselines must be violation-free (they see the same rules an LLM does).
"""

import numpy as np
import pytest

from finplan_arena.agents.baselines import (NaiveAgent, RuleBasedExpertAgent,
                                            TargetDateAgent)
from finplan_arena.runner import RunConfig, run_trial
from finplan_arena.scenarios.library import get_scenario

N_SEEDS = 60


@pytest.fixture(scope="module")
def paired_metrics():
    scenario = get_scenario("meridian")
    cfg = RunConfig(scenario="meridian", log_detail="summary")
    out = {}
    for cls in (RuleBasedExpertAgent, NaiveAgent):
        rows = []
        for seed in range(N_SEEDS):
            rec = run_trial(scenario, cls(), seed, cfg)
            rows.append(rec["metrics"])
        out[cls.name] = rows
    return out


def test_expert_beats_naive_on_ce_wealth(paired_metrics):
    expert = np.array([m["ce_wealth"] for m in paired_metrics["expert"]])
    naive = np.array([m["ce_wealth"] for m in paired_metrics["naive"]])
    win_rate = float(np.mean(expert > naive))
    assert win_rate >= 0.90, (
        f"expert only beats naive on {win_rate:.0%} of {N_SEEDS} seeds — "
        f"engine bug per spec §10")


def test_expert_pays_less_tax_than_naive(paired_metrics):
    expert = np.array([m["lifetime_taxes_real"] for m in paired_metrics["expert"]])
    naive = np.array([m["lifetime_taxes_real"] for m in paired_metrics["naive"]])
    assert float(np.mean(expert < naive)) >= 0.90


def test_baselines_are_violation_free(paired_metrics):
    for name, rows in paired_metrics.items():
        total = sum(m["violations"] for m in rows)
        assert total == 0, f"baseline '{name}' produced {total} violations"


def test_tdf_violation_free_all_scenarios():
    cfg = RunConfig(log_detail="summary")
    for scen_name in ("meridian", "summit", "foothill", "harbor"):
        scenario = get_scenario(scen_name)
        for cls in (TargetDateAgent, RuleBasedExpertAgent, NaiveAgent):
            rec = run_trial(scenario, cls(), 0, cfg)
            assert rec["metrics"]["violations"] == 0, (
                f"{cls.name} violated on {scen_name}: "
                f"{[v for lg in rec['logs'] for v in lg['violations']][:3]}")
