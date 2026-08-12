"""Return-path generators.

Seed discipline (the paired-trial contract):
    ``generate_path(seed, n_years)`` is a pure function of its arguments.
    Trial *k* uses the same path for every agent, so cross-agent comparisons
    difference out market luck. Nothing else in the engine draws randomness.

Two generators:

* ``BlockBootstrapGenerator`` (default) — samples overlapping 5-year blocks
  of *joint* historical annual observations (all asset classes + inflation
  from the same calendar years), preserving cross-asset correlation and
  within-block momentum/mean-reversion. Blocks are stitched until the
  horizon is covered.
* ``IIDLognormalGenerator`` — fits a multivariate normal to historical
  log(1+r) (jointly with log(1+inflation)) and samples IID years. For
  sensitivity checks; it deliberately destroys autocorrelation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from . import historical_data as hist

ASSET_KEYS = ("stocks_us", "stocks_intl", "bonds", "cash")


@dataclass
class YearReturns:
    """Joint nominal returns + realized inflation for one sim year."""

    stocks_us: float
    stocks_intl: float
    bonds: float
    cash: float
    inflation: float

    def asset_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in ASSET_KEYS}

    def to_dict(self) -> dict[str, float]:
        d = self.asset_dict()
        d["inflation"] = self.inflation
        return d


@dataclass
class ReturnPath:
    years: list[YearReturns]
    seed: int
    generator: str

    def __len__(self) -> int:
        return len(self.years)

    def __getitem__(self, i: int) -> YearReturns:
        return self.years[i]

    def __iter__(self) -> Iterator[YearReturns]:
        return iter(self.years)


def _matrix_to_path(m: np.ndarray, seed: int, generator: str) -> ReturnPath:
    years = [YearReturns(*(float(x) for x in row)) for row in m]
    return ReturnPath(years=years, seed=seed, generator=generator)


@dataclass
class BlockBootstrapGenerator:
    """Overlapping block bootstrap over the joint historical sample."""

    block_years: int = 5
    name: str = "block_bootstrap"

    def generate_path(self, seed: int, n_years: int) -> ReturnPath:
        data = hist.as_matrix()
        n_hist = data.shape[0]
        max_start = n_hist - self.block_years  # inclusive upper bound
        rng = np.random.default_rng(seed)
        rows: list[np.ndarray] = []
        while len(rows) < n_years:
            start = int(rng.integers(0, max_start + 1))
            rows.extend(data[start:start + self.block_years])
        m = np.array(rows[:n_years])
        return _matrix_to_path(m, seed, self.name)


@dataclass
class IIDLognormalGenerator:
    """IID multivariate lognormal fitted to historical log(1+r) moments.

    Inflation is sampled jointly (part of the same covariance matrix), so
    the inflation/return correlation structure survives even though serial
    correlation does not.
    """

    name: str = "iid_lognormal"
    _mean: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _cov: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def _fit(self) -> tuple[np.ndarray, np.ndarray]:
        if self._mean is None:
            log1p = np.log1p(hist.as_matrix())
            self._mean = log1p.mean(axis=0)
            self._cov = np.cov(log1p, rowvar=False)
        return self._mean, self._cov

    def generate_path(self, seed: int, n_years: int) -> ReturnPath:
        mean, cov = self._fit()
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(mean, cov, size=n_years,
                                        method="cholesky")
        m = np.expm1(draws)
        return _matrix_to_path(m, seed, self.name)


GENERATORS = {
    "block_bootstrap": BlockBootstrapGenerator,
    "iid_lognormal": IIDLognormalGenerator,
}


def make_generator(name: str, **kwargs):
    if name not in GENERATORS:
        raise ValueError(f"Unknown generator '{name}'. Options: {sorted(GENERATORS)}")
    return GENERATORS[name](**kwargs)
