"""Trial orchestration: scenario × agent × seeds.

Layout of a run directory (``runs/<name>/``):

* ``run.json``              — run configuration + agent/scenario identity
* ``trials/seed_00042.json`` — one file per trial: metrics + year logs
                               (``--log-detail summary`` trims the logs)
* ``summary.json``          — consolidated per-seed metrics (rebuilt at the
                               end of every run; cheap to regenerate)

Runs are **resumable**: a trial whose file already exists and parses is
skipped, so an interrupted LLM run continues where it left off (LLM
responses are additionally disk-cached by prompt hash inside the agent).

Seed discipline: trial *k* simulates the identical return path for every
agent. Nothing about the agent feeds the generator.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .agents.base import BaseAgent, Decision
from .agents.baselines import BASELINES, FourPercentDrawdownAgent
from .agents.llm_agent import LLMAgent, LLMConfig, MockLLMAgent
from .agents.validation import validate_decision
from .core.accounts import ContributionLimits
from .core.simulator import (SimConfig, build_observation, estimate_magi,
                             simulate_year)
from .core.life_events import years_until_both_reach
from .markets.returns import make_generator
from .scenarios.library import Scenario, get_scenario
from .scoring.comparison import RunData
from .scoring.metrics import MetricConfig, compute_trial_metrics

AgentFactory = Callable[[], BaseAgent]

#: YearLog fields kept when --log-detail=summary (full state traces off)
_SUMMARY_LOG_FIELDS = (
    "year_index", "calendar_year", "inflation_index_boy", "inflation_index_eoy",
    "all_retired", "violations", "shortfalls", "tax", "expenses",
    "consumption_real", "net_worth_nominal", "rationale", "rmd_forced",
    "roth_converted", "forced_liquidation", "unpaid_tax", "allocations_target",
)


@dataclass
class RunConfig:
    scenario: str = "meridian"
    agent: str = "expert"
    seeds: int = 300
    seed_start: int = 0
    workers: int = 8
    out_dir: Optional[str] = None
    generator: str = "block_bootstrap"
    block_years: int = 5
    log_detail: str = "full"            # "full" | "summary"
    agent_horizon: str = "active_then_drawdown"  # or "full" (agent to 95)
    llm_config_path: Optional[str] = None
    sim: SimConfig = field(default_factory=SimConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)

    def resolve_out_dir(self) -> str:
        if self.out_dir:
            return self.out_dir
        safe_agent = self.agent.replace(":", "_").replace("/", "_")
        return os.path.join("runs", f"{self.scenario}_{safe_agent}")


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------


def make_agent_factory(spec: str,
                       llm_config_path: Optional[str] = None) -> AgentFactory:
    """Agent spec → factory. Specs:

    * ``expert`` / ``tdf`` / ``naive``  — rule-based baselines
    * ``mock-llm``                       — deterministic canned LLM
    * ``llm:<path.json>``                — LLMAgent from a config file
    * ``claude-sonnet`` etc. (any other string) — LLMAgent on Anthropic with
      that model name (uses ANTHROPIC_API_KEY)
    """
    if spec in BASELINES:
        cls = BASELINES[spec]
        return lambda: cls()
    if spec == "mock-llm":
        return lambda: MockLLMAgent()
    if spec.startswith("llm:"):
        path = spec.split(":", 1)[1] or (llm_config_path or "")
        cfg = LLMConfig.from_file(path)
        return lambda: LLMAgent(cfg)
    if llm_config_path:
        cfg = LLMConfig.from_file(llm_config_path)
        return lambda: LLMAgent(cfg)
    # bare model name → Anthropic endpoint by default
    cfg = LLMConfig(provider="anthropic", model=spec)
    return lambda: LLMAgent(cfg)


AGENT_SPECS = sorted(BASELINES) + ["mock-llm", "llm:<config.json>",
                                   "<anthropic-model-name>"]


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def run_trial(scenario: Scenario, agent: BaseAgent, seed: int,
              cfg: RunConfig) -> dict[str, Any]:
    """Run one full trial. Returns the trial record (JSON-ready)."""
    gen_kwargs = {"block_years": cfg.block_years} \
        if cfg.generator == "block_bootstrap" else {}
    generator = make_generator(cfg.generator, **gen_kwargs)

    state = scenario.build_initial_state()
    total_years = years_until_both_reach(state, 95)
    path = generator.generate_path(seed=seed, n_years=total_years)

    agent.reset()
    drawdown: Optional[BaseAgent] = None
    logs: list[dict] = []
    agent_exceptions = 0

    for t in range(total_years):
        if cfg.agent_horizon == "full" or t < scenario.active_years:
            actor = agent
        else:
            if drawdown is None:
                drawdown = FourPercentDrawdownAgent()
                drawdown.reset()
            actor = drawdown

        obs = build_observation(state, cfg.sim, scenario.name)
        try:
            decision = actor.decide(obs)
        except Exception:
            agent_exceptions += 1
            decision = Decision.hold_prior(state)
            decision.rationale = ("fallback: agent raised "
                                  f"{traceback.format_exc(limit=1).splitlines()[-1]}")
        # validate here so violations are attributed to the agent, then the
        # engine applies the already-clipped decision
        limits = ContributionLimits.for_year(state.inflation_index,
                                             state.filing_status)
        vetted, violations = validate_decision(
            decision, state, limits, estimate_magi(state, decision, cfg.sim))
        state, log = simulate_year(state, vetted, path[t], cfg.sim,
                                   prevalidated=True)
        log.violations = [v.to_dict() for v in violations] + log.violations
        logs.append(log.to_dict())

    metrics = compute_trial_metrics(logs, state.to_dict(),
                                    state.goals.to_dict(), cfg.metrics)
    metrics["agent_exceptions"] = agent_exceptions
    if isinstance(agent, LLMAgent):
        metrics["llm_schema_fallbacks"] = agent.llm_failures

    record = {
        "schema": "finplan-arena/trial/1",
        "scenario": scenario.name,
        "agent": agent.name,
        "seed": seed,
        "generator": cfg.generator,
        "years_simulated": total_years,
        "active_years": (total_years if cfg.agent_horizon == "full"
                         else min(scenario.active_years, total_years)),
        "metrics": metrics,
        "final_state": state.to_dict(),
        "logs": logs if cfg.log_detail == "full" else
                [{k: lg[k] for k in _SUMMARY_LOG_FIELDS} for lg in logs],
    }
    return record


# ---------------------------------------------------------------------------
# Many trials (parallel, resumable)
# ---------------------------------------------------------------------------


def _trial_path(out_dir: str, seed: int) -> str:
    return os.path.join(out_dir, "trials", f"seed_{seed:05d}.json")


def _write_json(path: str, payload: dict) -> None:
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def run_many(cfg: RunConfig, agent_factory: Optional[AgentFactory] = None,
             scenario: Optional[Scenario] = None,
             progress: Optional[Callable[[str], None]] = print,
             ) -> str:
    """Run all requested seeds; returns the run directory."""
    scenario = scenario or get_scenario(cfg.scenario)
    agent_factory = agent_factory or make_agent_factory(cfg.agent,
                                                        cfg.llm_config_path)
    out_dir = cfg.resolve_out_dir()
    os.makedirs(os.path.join(out_dir, "trials"), exist_ok=True)

    probe = agent_factory()
    run_meta = {
        "schema": "finplan-arena/run/1",
        "scenario": scenario.name,
        "agent": probe.name,
        "agent_spec": cfg.agent,
        "generator": cfg.generator,
        "seeds_requested": [cfg.seed_start + i for i in range(cfg.seeds)],
        "log_detail": cfg.log_detail,
        "agent_horizon": cfg.agent_horizon,
        "sim_config": asdict(cfg.sim),
        "metric_config": asdict(cfg.metrics),
        "created_unix": int(time.time()),
    }
    _write_json(os.path.join(out_dir, "run.json"), run_meta)

    seeds = run_meta["seeds_requested"]
    todo = []
    for s in seeds:
        p = _trial_path(out_dir, s)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    json.load(f)
                continue  # done already (resume)
            except ValueError:
                pass  # corrupt → redo
        todo.append(s)

    if progress:
        progress(f"[{scenario.name} × {probe.name}] {len(seeds)} seeds, "
                 f"{len(seeds) - len(todo)} already done, running {len(todo)} "
                 f"with {cfg.workers} workers")

    def _one(seed: int) -> int:
        agent = agent_factory()
        record = run_trial(scenario, agent, seed, cfg)
        _write_json(_trial_path(out_dir, seed), record)
        return seed

    t0 = time.time()
    errors: list[str] = []
    if todo:
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
            futures = {pool.submit(_one, s): s for s in todo}
            done = 0
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append(f"seed {seed}: {e!r}")
                done += 1
                if progress and (done % 25 == 0 or done == len(todo)):
                    progress(f"  {done}/{len(todo)} trials "
                             f"({time.time() - t0:.1f}s)")
    if errors and progress:
        progress(f"  {len(errors)} trial(s) FAILED: " + "; ".join(errors[:5]))

    build_summary(out_dir)
    if progress:
        progress(f"  wrote {out_dir}/summary.json")
    return out_dir


def build_summary(out_dir: str) -> dict:
    """(Re)build summary.json from the trial files present."""
    trials_dir = os.path.join(out_dir, "trials")
    per_seed: dict[str, dict] = {}
    meta = {}
    run_path = os.path.join(out_dir, "run.json")
    if os.path.exists(run_path):
        with open(run_path) as f:
            meta = json.load(f)
    for fname in sorted(os.listdir(trials_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(trials_dir, fname)) as f:
            rec = json.load(f)
        per_seed[str(rec["seed"])] = rec["metrics"]
    summary = {
        "schema": "finplan-arena/summary/1",
        "scenario": meta.get("scenario"),
        "agent": meta.get("agent"),
        "n_trials": len(per_seed),
        "metrics_by_seed": per_seed,
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def load_run(out_dir: str) -> RunData:
    """Load a run directory into a RunData for comparison/reporting."""
    with open(os.path.join(out_dir, "summary.json")) as f:
        summary = json.load(f)
    return RunData(
        name=summary.get("agent") or os.path.basename(out_dir.rstrip("/")),
        scenario=summary.get("scenario") or "?",
        metrics_by_seed={int(k): v for k, v in summary["metrics_by_seed"].items()},
    )


def load_trial(out_dir: str, seed: int) -> dict:
    with open(_trial_path(out_dir, seed)) as f:
        return json.load(f)
