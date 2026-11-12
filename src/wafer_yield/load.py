"""Load the WM-811K wafer map dataset (LSWMD.pkl) into a clean DataFrame.

Raw columns in LSWMD.pkl:
    waferMap        2D array per wafer. 0 = no die, 1 = passing die, 2 = failing die
    dieSize         number of die sites on the wafer
    lotName         production lot ID, e.g. "lot1"
    waferIndex      wafer position within the lot (1 to 25)
    trianTestLabel  original train/test split label (typo is in the source data)
    failureType     defect pattern label, or empty if unlabeled
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FAILURE_TYPES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Near-full", "Random", "Scratch", "none",
]


def _unwrap_label(value) -> str | None:
    """Labels are stored as nested arrays like [['Center']] or as empty arrays.

    Returns the label string, or None when the wafer is unlabeled.
    """
    arr = np.asarray(value, dtype=object).ravel()
    if arr.size == 0:
        return None
    label = str(arr[0]).strip()
    return label or None


def _load_py2_pickle(path: str | Path):
    """Load a pickle written by Python 2. Text in those files is raw bytes, so
    it must be decoded as latin1 (the standard choice for numpy data).

    pandas' compatibility unpickler is tried first because it also maps old
    pandas class paths to their current names. If this pandas version does
    not expose it, the standard library unpickler is used instead."""
    with open(path, "rb") as f:
        try:
            from pandas.compat import pickle_compat
            return pickle_compat.Unpickler(f, encoding="latin1").load()
        except (ImportError, AttributeError, TypeError):
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read LSWMD.pkl.

    Two compatibility issues with this file:
    1. It was saved with an old pandas version, so newer pandas may fail with
       "No module named 'pandas.indexes'". The alias below points that old
       module path at its current location.
    2. It was saved from Python 2, so a plain read fails with
       UnicodeDecodeError. That case falls back to a latin1 read.
    """
    if "pandas.indexes" not in sys.modules:
        sys.modules["pandas.indexes"] = pd.core.indexes
    try:
        return pd.read_pickle(path)
    except UnicodeDecodeError:
        return _load_py2_pickle(path)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per wafer with typed columns and die counts.

    Output columns:
        lot, wafer_index, die_size, failure_type, labeled, split,
        n_die, n_pass, n_fail, wafer_yield, map_rows, map_cols
    """
    raw = raw.reset_index(drop=True)
    maps = raw["waferMap"].to_numpy()

    n_pass = np.fromiter((np.count_nonzero(m == 1) for m in maps), dtype=np.int64, count=len(maps))
    n_fail = np.fromiter((np.count_nonzero(m == 2) for m in maps), dtype=np.int64, count=len(maps))
    shapes = np.array([np.shape(m) for m in maps])

    df = pd.DataFrame({
        "lot": raw["lotName"].astype(str).to_numpy(),
        "wafer_index": pd.to_numeric(raw["waferIndex"], errors="coerce").astype("Int64"),
        "die_size": pd.to_numeric(raw["dieSize"], errors="coerce").to_numpy(),
        "failure_type": [_unwrap_label(v) for v in raw["failureType"]],
        "split": [_unwrap_label(v) for v in raw["trianTestLabel"]],
        "n_pass": n_pass,
        "n_fail": n_fail,
        "map_rows": shapes[:, 0],
        "map_cols": shapes[:, 1],
    })
    df["n_die"] = df["n_pass"] + df["n_fail"]
    df["labeled"] = df["failure_type"].notna()

    # Wafers with no dies at all cannot have a yield. Keep them out of yield math.
    df["wafer_yield"] = np.where(df["n_die"] > 0, df["n_pass"] / df["n_die"].where(df["n_die"] > 0), np.nan)
    return df


def load_wafers(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read and clean in one call. Also returns the raw wafer maps, indexed the
    same way as the cleaned table, for later phases (Pareto, classifier)."""
    raw = read_raw(path)
    maps = raw["waferMap"].reset_index(drop=True)
    return clean(raw), maps