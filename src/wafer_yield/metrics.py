"""Yield metrics at the wafer, lot, and dataset level.

Definitions used throughout the project:
    wafer yield  = passing dies / tested dies on one wafer
    lot yield    = total passing dies / total tested dies across the lot
                   (die-weighted, so a large wafer counts more than a small one)
    overall yield = total passing dies / total tested dies in the dataset
"""

from __future__ import annotations

import pandas as pd


def lot_yield(wafers: pd.DataFrame) -> pd.DataFrame:
    """One row per lot, sorted by lot yield ascending (worst lots first)."""
    tested = wafers[wafers["n_die"] > 0]
    lots = (
        tested.groupby("lot", sort=False)
        .agg(
            n_wafers=("wafer_yield", "size"),
            n_die=("n_die", "sum"),
            n_pass=("n_pass", "sum"),
            n_fail=("n_fail", "sum"),
            mean_wafer_yield=("wafer_yield", "mean"),
            min_wafer_yield=("wafer_yield", "min"),
            n_labeled=("labeled", "sum"),
        )
        .reset_index()
    )
    lots["lot_yield"] = lots["n_pass"] / lots["n_die"]
    return lots.sort_values("lot_yield").reset_index(drop=True)


def label_distribution(wafers: pd.DataFrame) -> pd.DataFrame:
    """Count and share of each defect label among labeled wafers, plus the
    dies lost to each label. Sorted by wafer count."""
    labeled = wafers[wafers["labeled"]]
    dist = (
        labeled.groupby("failure_type")
        .agg(n_wafers=("n_die", "size"), dies_lost=("n_fail", "sum"),
             mean_wafer_yield=("wafer_yield", "mean"))
        .sort_values("n_wafers", ascending=False)
        .reset_index()
    )
    dist["share_of_labeled"] = dist["n_wafers"] / dist["n_wafers"].sum()
    return dist


def summary(wafers: pd.DataFrame, lots: pd.DataFrame) -> dict:
    """Headline numbers for the README and the weekly review."""
    tested = wafers[wafers["n_die"] > 0]
    return {
        "n_wafers": int(len(wafers)),
        "n_wafers_tested": int(len(tested)),
        "n_wafers_no_die": int((wafers["n_die"] == 0).sum()),
        "n_lots": int(wafers["lot"].nunique()),
        "n_labeled": int(wafers["labeled"].sum()),
        "share_labeled": float(wafers["labeled"].mean()),
        "n_die_total": int(tested["n_die"].sum()),
        "overall_yield": float(tested["n_pass"].sum() / tested["n_die"].sum()),
        "median_wafer_yield": float(tested["wafer_yield"].median()),
        "median_lot_yield": float(lots["lot_yield"].median()),
        "wafers_per_lot_median": float(lots["n_wafers"].median()),
        "distinct_map_shapes": int(wafers[["map_rows", "map_cols"]].drop_duplicates().shape[0]),
    }
