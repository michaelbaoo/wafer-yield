"""Tests for phase 2 SPC. Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_yield.spc import lot_number, order_lots, p_charts, spc_summary  # noqa: E402


def _lots(n_die, n_fail, names=None):
    names = names or [f"lot{i + 1}" for i in range(len(n_die))]
    return pd.DataFrame({"lot": names, "n_die": n_die, "n_fail": n_fail,
                         "n_wafers": 25})


def test_lot_number_parsing():
    assert lot_number("lot1234") == 1234
    assert lot_number("LOT_77") == 77
    assert np.isnan(lot_number("pilot"))


def test_order_is_numeric_not_text():
    df = order_lots(_lots([10, 10, 10, 10], [1, 1, 1, 1], ["lot10", "lot2", "abc", "lot1"]))
    assert list(df["lot"]) == ["lot1", "lot2", "lot10", "abc"]
    assert df["lot_num_missing"].tolist() == [False, False, False, True]


def test_standard_limits_by_hand():
    # Two lots: 100 dies with 10 fails, 300 dies with 10 fails.
    # p_bar = 20 / 400 = 0.05
    # sigma for lot 1 = sqrt(0.05 * 0.95 / 100) = 0.021794
    df, params = p_charts(order_lots(_lots([100, 300], [10, 10])))
    assert params["p_bar"] == pytest.approx(0.05)
    s1 = np.sqrt(0.05 * 0.95 / 100)
    assert df.loc[0, "ucl_std"] == pytest.approx(0.05 + 3 * s1)
    assert df.loc[0, "lcl_std"] == pytest.approx(max(0.0, 0.05 - 3 * s1))


def test_laney_close_to_standard_for_pure_binomial():
    # Every lot has the same true fail rate, so variation is only sampling noise.
    rng = np.random.default_rng(0)
    n = rng.integers(20_000, 40_000, size=5_000)
    fails = rng.binomial(n, 0.07)
    df, params = p_charts(order_lots(_lots(n, fails)))
    assert params["sigma_z"] == pytest.approx(1.0, abs=0.05)
    # Expected high-side false alarms: 5000 * 0.00135 = 6.75
    assert df["ooc_high_laney"].sum() < 25


def test_laney_widens_for_overdispersion():
    # True fail rate varies lot to lot, as in real production.
    rng = np.random.default_rng(1)
    n = rng.integers(20_000, 40_000, size=5_000)
    true_p = rng.beta(7, 93, size=5_000)  # mean 0.07, real lot-to-lot spread
    fails = rng.binomial(n, true_p)
    df, params = p_charts(order_lots(_lots(n, fails)))
    stats = spc_summary(df, params)
    assert params["sigma_z"] > 5
    # The standard chart flags a large share of lots; Laney flags far fewer.
    assert stats["ooc_high_std_share"] > 0.2
    assert stats["ooc_high_laney"] < stats["ooc_high_std"] / 5


def test_excess_fails_sum_to_zero():
    rng = np.random.default_rng(2)
    n = rng.integers(1_000, 2_000, size=300)
    fails = rng.binomial(n, 0.1)
    df, _ = p_charts(order_lots(_lots(n, fails)))
    # Excess over the pooled average must net out across all lots.
    assert df["excess_fails"].sum() == pytest.approx(0.0, abs=1e-6)