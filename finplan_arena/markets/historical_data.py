"""Embedded historical annual return series, 1928-2024.

Sources (annual total-return series in the style of):
* US large-cap stocks (S&P 500 incl. dividends), 10-year US Treasuries,
  and 3-month T-bills: A. Damodaran, "Historical Returns on Stocks, Bonds
  and Bills: 1928-2024" (NYU Stern), which itself builds on S&P/FRED data.
* CPI: BLS CPI-U, December-over-December (as tabulated alongside the
  Shiller/Damodaran series).
* International developed stocks: MSCI EAFE (USD, gross) from its 1970
  inception. Pre-1970 values are spliced as US stocks minus a 150 bps
  spread (a documented, deliberately crude splice — flagged inline).

NOTE ON PRECISION: the arrays below are transcriptions of the published
series and may differ from the source by small amounts in individual years.
The harness's statistical machinery (block bootstrap over joint annual
observations) only needs the joint distribution to be historically
plausible; refresh from the sources above if exact fidelity matters.

All values are NOMINAL annual total returns (decimals, 0.10 == +10%).
Real returns are derived where needed via (1+nominal)/(1+cpi) - 1.
Inflation is CPI December-over-December (decimal).
"""

from __future__ import annotations

import numpy as np

FIRST_YEAR = 1928
LAST_YEAR = 2024

# fmt: off
#: S&P 500 total return (nominal), 1928-2024
STOCKS_US = [
    0.4381, -0.0830, -0.2512, -0.4384, -0.0864, 0.4998, -0.0119, 0.4674,
    0.3194, -0.3534, 0.2928, -0.0110, -0.1067, -0.1277, 0.1917, 0.2506,
    0.1903, 0.3582, -0.0843, 0.0520, 0.0570, 0.1830, 0.3081, 0.2368,
    0.1815, -0.0121, 0.5256, 0.3260, 0.0744, -0.1046, 0.4372, 0.1206,
    0.0034, 0.2664, -0.0881, 0.2261, 0.1642, 0.1240, -0.0997, 0.2380,
    0.1081, -0.0824, 0.0356, 0.1422, 0.1876, -0.1431, -0.2590, 0.3700,
    0.2383, -0.0698, 0.0651, 0.1852, 0.3174, -0.0470, 0.2042, 0.2234,
    0.0615, 0.3124, 0.1849, 0.0581, 0.1654, 0.3148, -0.0306, 0.3023,
    0.0749, 0.0997, 0.0133, 0.3720, 0.2268, 0.3310, 0.2834, 0.2089,
    -0.0903, -0.1185, -0.2197, 0.2836, 0.1074, 0.0483, 0.1561, 0.0548,
    -0.3655, 0.2594, 0.1482, 0.0210, 0.1589, 0.3215, 0.1352, 0.0138,
    0.1177, 0.2161, -0.0423, 0.3121, 0.1802, 0.2847, -0.1804, 0.2606,
    0.2488,
]

#: 10-year US Treasury total return (nominal), 1928-2024
BONDS_10Y = [
    0.0084, 0.0420, 0.0454, -0.0256, 0.0879, 0.0186, 0.0796, 0.0447,
    0.0502, 0.0138, 0.0421, 0.0441, 0.0540, -0.0202, 0.0229, 0.0249,
    0.0258, 0.0380, 0.0313, 0.0092, 0.0195, 0.0466, 0.0043, -0.0030,
    0.0227, 0.0414, 0.0329, -0.0134, -0.0226, 0.0680, -0.0210, -0.0265,
    0.1164, 0.0206, 0.0569, 0.0168, 0.0373, 0.0072, 0.0291, -0.0158,
    0.0327, -0.0501, 0.1675, 0.0979, 0.0282, 0.0366, 0.0199, 0.0361,
    0.1598, 0.0129, -0.0078, 0.0067, -0.0299, 0.0820, 0.3281, 0.0320,
    0.1373, 0.2571, 0.2428, -0.0496, 0.0822, 0.1769, 0.0624, 0.1500,
    0.0936, 0.1421, -0.0804, 0.2348, 0.0143, 0.0994, 0.1492, -0.0825,
    0.1666, 0.0557, 0.1512, 0.0038, 0.0449, 0.0287, 0.0196, 0.1021,
    0.2010, -0.1112, 0.0846, 0.1604, 0.0297, -0.0910, 0.1075, 0.0128,
    0.0069, 0.0280, -0.0002, 0.0964, 0.1133, -0.0442, -0.1783, 0.0388,
    -0.0164,
]

#: 3-month T-bill return (nominal), 1928-2024
TBILLS = [
    0.0308, 0.0316, 0.0455, 0.0231, 0.0107, 0.0096, 0.0028, 0.0017,
    0.0017, 0.0028, 0.0007, 0.0005, 0.0004, 0.0013, 0.0034, 0.0038,
    0.0038, 0.0038, 0.0038, 0.0060, 0.0105, 0.0112, 0.0120, 0.0152,
    0.0172, 0.0189, 0.0094, 0.0172, 0.0262, 0.0322, 0.0177, 0.0339,
    0.0287, 0.0235, 0.0277, 0.0316, 0.0355, 0.0395, 0.0486, 0.0429,
    0.0534, 0.0667, 0.0639, 0.0433, 0.0406, 0.0704, 0.0785, 0.0579,
    0.0498, 0.0526, 0.0718, 0.1005, 0.1139, 0.1404, 0.1060, 0.0862,
    0.0954, 0.0747, 0.0597, 0.0578, 0.0667, 0.0811, 0.0750, 0.0538,
    0.0343, 0.0300, 0.0425, 0.0549, 0.0501, 0.0506, 0.0478, 0.0464,
    0.0582, 0.0339, 0.0160, 0.0101, 0.0137, 0.0315, 0.0473, 0.0435,
    0.0137, 0.0015, 0.0014, 0.0005, 0.0009, 0.0006, 0.0003, 0.0005,
    0.0032, 0.0093, 0.0194, 0.0206, 0.0035, 0.0005, 0.0202, 0.0507,
    0.0500,
]

#: CPI-U inflation, December over December (decimal), 1928-2024
CPI = [
    -0.0170, 0.0060, -0.0640, -0.0930, -0.1030, 0.0080, 0.0150, 0.0300,
    0.0140, 0.0290, -0.0280, 0.0000, 0.0070, 0.0990, 0.0900, 0.0300,
    0.0230, 0.0220, 0.1810, 0.0880, 0.0300, -0.0210, 0.0590, 0.0600,
    0.0080, 0.0070, -0.0070, 0.0040, 0.0300, 0.0290, 0.0180, 0.0170,
    0.0140, 0.0070, 0.0130, 0.0160, 0.0100, 0.0190, 0.0350, 0.0300,
    0.0470, 0.0620, 0.0560, 0.0330, 0.0340, 0.0870, 0.1230, 0.0690,
    0.0490, 0.0670, 0.0900, 0.1330, 0.1250, 0.0890, 0.0380, 0.0380,
    0.0390, 0.0380, 0.0110, 0.0440, 0.0440, 0.0460, 0.0610, 0.0310,
    0.0290, 0.0270, 0.0270, 0.0250, 0.0330, 0.0170, 0.0160, 0.0270,
    0.0340, 0.0160, 0.0240, 0.0190, 0.0330, 0.0340, 0.0250, 0.0410,
    0.0010, 0.0270, 0.0150, 0.0300, 0.0170, 0.0150, 0.0080, 0.0070,
    0.0210, 0.0210, 0.0190, 0.0230, 0.0140, 0.0700, 0.0650, 0.0340,
    0.0290,
]

#: MSCI EAFE (USD) total return from 1970; 1928-1969 spliced as US - 150bps.
#: (The splice keeps the joint bootstrap well-defined over the full sample
#: at the cost of understating true pre-1970 diversification benefit.)
_EAFE_1970_ON = [
    -0.1117, 0.2959, 0.3635, -0.1492, -0.2316, 0.3539, 0.0254, 0.1806,
    0.3261, 0.0475, 0.2258, -0.0226, -0.0186, 0.2369, 0.0738, 0.5616,
    0.6944, 0.2463, 0.2827, 0.1054, -0.2345, 0.1220, -0.1223, 0.3257,
    0.0778, 0.1121, 0.0605, 0.0178, 0.2000, 0.2696, -0.1421, -0.2144,
    -0.1594, 0.3859, 0.2025, 0.1354, 0.2634, 0.1117, -0.4338, 0.3178,
    0.0775, -0.1214, 0.1732, 0.2278, -0.0490, -0.0081, 0.0100, 0.2503,
    -0.1379, 0.2201, 0.0782, 0.1126, -0.1445, 0.1824, 0.0382,
]

_SPLICE_SPREAD = 0.015
STOCKS_INTL = [r - _SPLICE_SPREAD for r in STOCKS_US[: 1970 - FIRST_YEAR]] + _EAFE_1970_ON
# fmt: on

YEARS = list(range(FIRST_YEAR, LAST_YEAR + 1))

_N = len(YEARS)
for _name, _series in (("STOCKS_US", STOCKS_US), ("STOCKS_INTL", STOCKS_INTL),
                       ("BONDS_10Y", BONDS_10Y), ("TBILLS", TBILLS), ("CPI", CPI)):
    if len(_series) != _N:  # pragma: no cover - guards transcription errors
        raise RuntimeError(f"historical series {_name} has {len(_series)} entries, expected {_N}")


def as_matrix() -> np.ndarray:
    """(n_years, 5) matrix of joint annual observations, columns ordered as
    [stocks_us, stocks_intl, bonds, cash(t-bills), inflation]."""
    return np.column_stack([
        np.array(STOCKS_US), np.array(STOCKS_INTL), np.array(BONDS_10Y),
        np.array(TBILLS), np.array(CPI),
    ])


COLUMNS = ("stocks_us", "stocks_intl", "bonds", "cash", "inflation")


def real_return(nominal: float, inflation: float) -> float:
    return (1.0 + nominal) / (1.0 + inflation) - 1.0


def summary_stats() -> dict[str, dict[str, float]]:
    """Arithmetic mean / stdev per column (nominal), for sanity checks."""
    m = as_matrix()
    return {
        c: {"mean": float(m[:, i].mean()), "std": float(m[:, i].std(ddof=1))}
        for i, c in enumerate(COLUMNS)
    }
