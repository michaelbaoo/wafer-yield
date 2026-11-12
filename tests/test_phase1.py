"""Tests for phase 1. Run with:  python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from wafer_yield.load import _unwrap_label, clean, read_raw  # noqa: E402
from wafer_yield.metrics import label_distribution, lot_yield, summary  # noqa: E402
import make_synthetic  # noqa: E402


def _raw(maps, lots, labels):
    return pd.DataFrame({
        "waferMap": maps,
        "dieSize": [float((m > 0).sum()) for m in maps],
        "lotName": lots,
        "waferIndex": [float(i + 1) for i in range(len(maps))],
        "trianTestLabel": [np.array([], dtype=object)] * len(maps),
        "failureType": labels,
    })


def test_unwrap_label():
    assert _unwrap_label(np.array([["Scratch"]], dtype=object)) == "Scratch"
    assert _unwrap_label(np.array([], dtype=object)) is None
    assert _unwrap_label([["none"]]) == "none"


def test_wafer_yield_counts():
    m = np.array([[0, 1, 1], [1, 2, 1], [0, 2, 0]])  # 4 pass, 2 fail, 3 empty sites
    df = clean(_raw([m], ["lot1"], [np.array([["Loc"]], dtype=object)]))
    row = df.iloc[0]
    assert (row.n_pass, row.n_fail, row.n_die) == (4, 2, 6)
    assert row.wafer_yield == pytest.approx(4 / 6)
    assert bool(row.labeled) and row.failure_type == "Loc"


def test_lot_yield_is_die_weighted():
    big = np.ones((10, 10), dtype=int)          # 100 dies, all pass
    small = np.array([[1, 2], [2, 2]])           # 4 dies, 1 pass
    empty = np.array([], dtype=object)
    df = clean(_raw([big, small], ["A", "A"], [empty, empty]))
    lots = lot_yield(df)
    # Die-weighted: 101 / 104. A plain average of wafer yields would be 0.625.
    assert lots.loc[0, "lot_yield"] == pytest.approx(101 / 104)
    assert lots.loc[0, "mean_wafer_yield"] == pytest.approx((1.0 + 0.25) / 2)


def test_empty_wafer_excluded_from_yield():
    empty_map = np.zeros((3, 3), dtype=int)
    ok_map = np.array([[1, 2]])
    e = np.array([], dtype=object)
    df = clean(_raw([empty_map, ok_map], ["A", "B"], [e, e]))
    assert np.isnan(df.loc[0, "wafer_yield"])
    lots = lot_yield(df)
    assert set(lots["lot"]) == {"B"}


def test_reads_python2_style_pickle(tmp_path):
    # Hand-built protocol 2 pickle holding a Python 2 byte string with a
    # non-ASCII byte (0x9a), the same thing that breaks on LSWMD.pkl.
    # Opcodes: PROTO 2, BINSTRING of length 2, STOP.
    p = tmp_path / "py2.pkl"
    p.write_bytes(b"\x80\x02T\x02\x00\x00\x00\x9a\x01.")
    assert read_raw(p) == "\x9a\x01"


def test_full_pipeline_on_synthetic():
    raw = make_synthetic.build(n_lots=5, seed=1)
    df = clean(raw)
    lots = lot_yield(df)
    labels = label_distribution(df)
    stats = summary(df, lots)
    assert stats["n_wafers"] == len(raw)
    assert stats["n_wafers_no_die"] == 1
    assert stats["n_lots"] == 6  # 5 lots plus the empty one
    assert 0 < stats["overall_yield"] < 1
    assert labels["share_of_labeled"].sum() == pytest.approx(1.0)
    # Totals reconcile: label table dies lost equals fails on labeled wafers.
    assert labels["dies_lost"].sum() == df.loc[df.labeled, "n_fail"].sum()