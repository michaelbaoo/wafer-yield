"""Tests for phase 3 Pareto. Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_yield.pareto import (  # noqa: E402
    ks_statistic, labeled_split, pattern_mix_by_flag, pattern_pareto,
)


def _wafers():
    # lot A: two Scratch wafers losing 10 each, one none losing 50
    # lot B: one Random wafer losing 300, one unlabeled losing 40
    return pd.DataFrame({
        "lot":          ["A", "A", "A", "B", "B"],
        "failure_type": ["Scratch", "Scratch", "none", "Random", None],
        "labeled":      [True, True, True, True, False],
        "n_fail":       [10, 10, 50, 300, 40],
        "n_die":        [100, 100, 100, 400, 100],
    }).assign(wafer_yield=lambda d: 1 - d.n_fail / d.n_die)


def test_pareto_ranks_by_dies_lost_not_count():
    p = pattern_pareto(_wafers())
    # Scratch is more frequent (2 wafers) but Random loses more dies (300 vs 20).
    assert list(p["failure_type"]) == ["Random", "Scratch"]
    assert p.loc[0, "rank_count"] == 2 and p.loc[1, "rank_count"] == 1
    assert p.loc[1, "dies_lost_per_wafer"] == 10
    assert p["cum_share_dies_lost"].iloc[-1] == pytest.approx(1.0)
    assert "none" not in set(p["failure_type"])


def test_labeled_split_sums_to_total():
    s = labeled_split(_wafers())
    assert s["unlabeled"]["dies_lost"] == 40
    assert s["labeled_none"]["dies_lost"] == 50
    assert s["labeled_patterned"]["dies_lost"] == 320
    assert sum(v["share"] for v in s.values()) == pytest.approx(1.0)


def test_ks_statistic():
    rng = np.random.default_rng(0)
    same = ks_statistic(rng.normal(size=5000), rng.normal(size=5000))
    shifted = ks_statistic(rng.normal(size=5000), rng.normal(2, 1, size=5000))
    assert same < 0.05
    assert shifted > 0.6
    assert ks_statistic(np.array([0.0, 1.0]), np.array([5.0, 6.0])) == 1.0


def test_pattern_mix_lift():
    lots = pd.DataFrame({"lot": ["A", "B"], "ooc_high_laney": [False, True]})
    mix = pattern_mix_by_flag(_wafers(), lots).set_index("failure_type")
    # Flagged lot B has only Random among labeled wafers, so Random share = 1.
    assert mix.loc["Random", "share_flagged"] == 1.0
    assert mix.loc["Random", "n_other"] == 0
    assert mix.loc["Scratch", "share_other"] == pytest.approx(2 / 3)
    assert mix["share_flagged"].sum() == pytest.approx(1.0)