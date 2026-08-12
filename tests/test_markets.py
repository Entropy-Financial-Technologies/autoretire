"""Return generators: seed discipline, joint sampling, shapes."""

import numpy as np
import pytest

from finplan_arena.markets import historical_data as hist
from finplan_arena.markets.returns import (BlockBootstrapGenerator,
                                           IIDLognormalGenerator,
                                           make_generator)


def test_historical_series_aligned():
    m = hist.as_matrix()
    assert m.shape == (97, 5)  # 1928..2024 inclusive
    assert hist.YEARS[0] == 1928 and hist.YEARS[-1] == 2024


def test_historical_sanity_moments():
    s = hist.summary_stats()
    assert 0.09 < s["stocks_us"]["mean"] < 0.14
    assert 0.15 < s["stocks_us"]["std"] < 0.25
    assert 0.02 < s["inflation"]["mean"] < 0.045
    assert s["cash"]["std"] < s["bonds"]["std"] < s["stocks_us"]["std"]


def test_block_bootstrap_deterministic_and_shaped():
    g = BlockBootstrapGenerator(block_years=5)
    p1 = g.generate_path(seed=42, n_years=59)
    p2 = g.generate_path(seed=42, n_years=59)
    assert len(p1) == 59
    assert [y.to_dict() for y in p1] == [y.to_dict() for y in p2]
    p3 = g.generate_path(seed=43, n_years=59)
    assert [y.to_dict() for y in p1] != [y.to_dict() for y in p3]


def test_blocks_are_joint_historical_rows():
    """Every sampled year must be an actual historical row (all five series
    from the same calendar year) — that's what preserves correlations."""
    g = BlockBootstrapGenerator(block_years=5)
    path = g.generate_path(seed=7, n_years=40)
    data = hist.as_matrix()
    rows = {tuple(np.round(r, 10)) for r in data}
    for y in path:
        row = (round(y.stocks_us, 10), round(y.stocks_intl, 10),
               round(y.bonds, 10), round(y.cash, 10), round(y.inflation, 10))
        assert row in rows


def test_iid_lognormal_moments_close_to_history():
    g = IIDLognormalGenerator()
    path = g.generate_path(seed=1, n_years=20_000)
    us = np.array([y.stocks_us for y in path])
    infl = np.array([y.inflation for y in path])
    s = hist.summary_stats()
    assert abs(np.log1p(us).mean() - np.log1p(hist.as_matrix()[:, 0]).mean()) < 0.01
    assert abs(infl.mean() - s["inflation"]["mean"]) < 0.01


def test_make_generator_registry():
    assert isinstance(make_generator("block_bootstrap", block_years=3),
                      BlockBootstrapGenerator)
    assert isinstance(make_generator("iid_lognormal"), IIDLognormalGenerator)
    with pytest.raises(ValueError):
        make_generator("crystal_ball")
