"""End-to-end smoke test (spec §10): a mock LLM through the full stack —
runner → trial files → summary → paired comparison → report — plus resume
behavior and the CLI entry point."""

import json
import os

from finplan_arena.cli import main as cli_main
from finplan_arena.runner import (RunConfig, load_run, load_trial,
                                  make_agent_factory, run_many)
from finplan_arena.scoring.comparison import (compare_runs,
                                              format_comparison_text,
                                              paired_comparison)


def _run(tmp_path, agent, seeds=3, **kw):
    out = str(tmp_path / f"run_{agent.replace(':', '_')}")
    cfg = RunConfig(scenario="meridian", agent=agent, seeds=seeds,
                    workers=2, out_dir=out, log_detail="summary", **kw)
    run_many(cfg, progress=None)
    return out


def test_mock_llm_end_to_end(tmp_path):
    out_mock = _run(tmp_path, "mock-llm")
    out_naive = _run(tmp_path, "naive")

    # trial files exist and carry the full record
    rec = load_trial(out_mock, 0)
    assert rec["agent"] == "mock-llm"
    assert rec["metrics"]["ce_wealth"] > 0
    assert rec["years_simulated"] == 59          # both adults to 95
    assert len(rec["logs"]) == 59
    assert rec["metrics"].get("llm_schema_fallbacks", 0) == 0

    # paired comparison runs and is seed-matched
    a, b = load_run(out_mock), load_run(out_naive)
    p = paired_comparison(a, b, "ce_wealth")
    assert p.n_seeds == 3

    payload = compare_runs([a, b], reference="naive")
    text = format_comparison_text(payload)
    assert "median CE wealth" in text and "mock-llm" in text


def test_resume_skips_done_trials(tmp_path):
    out = str(tmp_path / "resumable")
    cfg = RunConfig(scenario="meridian", agent="naive", seeds=2, workers=1,
                    out_dir=out, log_detail="summary")
    run_many(cfg, progress=None)
    stamp = os.path.getmtime(os.path.join(out, "trials", "seed_00000.json"))
    cfg2 = RunConfig(scenario="meridian", agent="naive", seeds=3, workers=1,
                     out_dir=out, log_detail="summary")
    run_many(cfg2, progress=None)                # adds seed 2 only
    assert os.path.getmtime(os.path.join(out, "trials", "seed_00000.json")) == stamp
    assert load_run(out).seeds == {0, 1, 2}


def test_cli_run_compare_report(tmp_path, capsys):
    out1 = str(tmp_path / "cli_expert")
    out2 = str(tmp_path / "cli_naive")
    assert cli_main(["run", "--scenario", "meridian", "--agent", "expert",
                     "--seeds", "2", "--workers", "1", "--out", out1,
                     "--log-detail", "summary"]) == 0
    assert cli_main(["run", "--scenario", "meridian", "--agent", "naive",
                     "--seeds", "2", "--workers", "1", "--out", out2,
                     "--log-detail", "summary"]) == 0
    assert cli_main(["compare", out1, out2]) == 0
    text = capsys.readouterr().out
    assert "Paired differences vs 'expert'" in text

    report_md = str(tmp_path / "report.md")
    assert cli_main(["report", out1, "--out", report_md, "--worst", "1"]) == 0
    body = open(report_md).read()
    assert "Aggregate" in body and "post-mortem" in body


def test_agent_factory_specs():
    assert make_agent_factory("expert")().name == "expert"
    assert make_agent_factory("mock-llm")().name == "mock-llm"


def test_paired_seeds_share_return_paths(tmp_path):
    """The paired design contract: different agents, same seed → identical
    return sequence in the logs."""
    out_a = _run(tmp_path, "expert", seeds=1)
    out_b = _run(tmp_path, "naive", seeds=1)
    ra, rb = load_trial(out_a, 0), load_trial(out_b, 0)
    infl_a = [lg["inflation_index_eoy"] for lg in ra["logs"]]
    infl_b = [lg["inflation_index_eoy"] for lg in rb["logs"]]
    assert infl_a == infl_b