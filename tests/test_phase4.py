"""Tests for phase 4 features and classifier. Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wafer_yield.classify import (  # noqa: E402
    evaluate, make_model, pareto_with_predictions, split_by_lot,
)
from wafer_yield.features import N_FEATURES, build_matrix, feature_names, wafer_features  # noqa: E402
from make_synthetic import LABELS, make_map  # noqa: E402

NAMES = feature_names()
COL = {n: i for i, n in enumerate(NAMES)}


def test_feature_length_and_empty_wafer():
    assert len(NAMES) == N_FEATURES == 85
    assert np.all(wafer_features(np.zeros((10, 10))) == 0)


def test_rings_separate_center_from_edge():
    rng = np.random.default_rng(0)
    center = wafer_features(make_map(rng, 40, 0.0, "Center"))
    edge = wafer_features(make_map(rng, 40, 0.0, "Edge-Ring"))
    assert center[COL["ring_0"]] > 0.5 and center[COL["ring_4"]] < 0.05
    assert edge[COL["ring_4"]] > 0.3 and edge[COL["ring_0"]] < 0.05
    assert edge[COL["edge_ratio"]] > center[COL["edge_ratio"]]


def test_scratch_is_elongated_and_loc_is_not():
    rng = np.random.default_rng(1)
    scratch = wafer_features(make_map(rng, 40, 0.0, "Scratch"))
    loc = wafer_features(make_map(rng, 40, 0.0, "Loc"))
    assert scratch[COL["largest_cluster_elong"]] > 0.9
    assert loc[COL["largest_cluster_elong"]] < 0.5


def test_fail_rate_matches_counts():
    m = np.array([[0, 1, 2], [1, 2, 1], [0, 1, 0]])
    f = wafer_features(m)
    assert f[COL["fail_rate"]] == pytest.approx(2 / 6)
    assert f[COL["die_count"]] == 6


def test_split_has_no_shared_lots():
    lots = np.repeat([f"lot{i}" for i in range(100)], 20)
    tr, te = split_by_lot(lots)
    assert not set(lots[tr]) & set(lots[te])
    assert len(tr) + len(te) == len(lots)


def test_classifier_learns_synthetic_patterns():
    rng = np.random.default_rng(2)
    n_lots, per_lot = 120, 20
    maps, y, lots = [], [], []
    for lot in range(n_lots):
        size = int(rng.choice([26, 32, 40]))
        for _ in range(per_lot):
            p = str(rng.choice(LABELS))
            maps.append(make_map(rng, size, rng.uniform(0.01, 0.05), p))
            y.append(p)
            lots.append(f"lot{lot}")
    X = build_matrix(maps, progress_every=0)
    y, lots = np.array(y), np.array(lots)
    tr, te = split_by_lot(lots)
    model = make_model().fit(X[tr], y[tr])
    report, cm, macro = evaluate(y[te], model.predict(X[te]), sorted(LABELS))
    # Synthetic patterns are clean, so this checks the plumbing, not real accuracy.
    assert macro > 0.85
    assert cm.to_numpy().sum() == len(te)


def test_pareto_uses_predictions_only_for_unlabeled():
    index = pd.DataFrame({
        "failure_type": ["Center", None, None],
        "labeled": [True, False, False],
        "n_fail": [10, 20, 30],
    })
    pred = np.array(["Scratch", "Donut", "Donut"])   # first is ignored: wafer is labeled
    conf = np.array([0.99, 0.95, 0.50])
    p = pareto_with_predictions(index, pred, conf, min_conf=0.8).set_index("pattern")
    assert p.loc["Center", "dies_lost"] == 10
    assert p.loc["Donut", "dies_lost"] == 20
    assert p.loc["uncertain", "dies_lost"] == 30
    assert "Scratch" not in p.index
    assert p["share_all_dies_lost"].sum() == pytest.approx(1.0)