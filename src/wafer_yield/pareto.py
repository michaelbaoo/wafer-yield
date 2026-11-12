"""Defect Pareto analysis.

Three questions:
1. Which defect patterns cost the most dies? Ranked by dies lost, with
   wafer count alongside, because the two rankings can disagree.
2. Do labeled wafers look like unlabeled ones? Only ~21% of wafers carry a
   pattern label. If the labeled set is not representative, the Pareto
   describes the labeled set, not the whole dataset.
3. Which patterns are over-represented in the lots the Laney chart flagged?
   This links the SPC result (phase 2) to likely causes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NO_PATTERN = "none"


def pattern_pareto(wafers: pd.DataFrame) -> pd.DataFrame:
    """Pareto of patterned wafers (labeled, excluding 'none'), sorted by
    dies lost. Also gives each pattern's rank by wafer count."""
    d = wafers[wafers["labeled"] & (wafers["failure_type"] != NO_PATTERN)]
    p = (
        d.groupby("failure_type")
        .agg(n_wafers=("n_fail", "size"), dies_lost=("n_fail", "sum"),
             mean_wafer_yield=("wafer_yield", "mean"))
        .reset_index()
        .sort_values("dies_lost", ascending=False)
        .reset_index(drop=True)
    )
    p["dies_lost_per_wafer"] = p["dies_lost"] / p["n_wafers"]
    p["share_dies_lost"] = p["dies_lost"] / p["dies_lost"].sum()
    p["cum_share_dies_lost"] = p["share_dies_lost"].cumsum()
    p["share_wafers"] = p["n_wafers"] / p["n_wafers"].sum()
    p["rank_dies_lost"] = np.arange(1, len(p) + 1)
    p["rank_count"] = p["n_wafers"].rank(ascending=False, method="min").astype(int)
    return p


def labeled_split(wafers: pd.DataFrame) -> dict:
    """Total dies lost split into unlabeled, labeled 'none', and patterned."""
    lab = wafers["labeled"]
    none = lab & (wafers["failure_type"] == NO_PATTERN)
    total = float(wafers["n_fail"].sum())
    parts = {
        "unlabeled": float(wafers.loc[~lab, "n_fail"].sum()),
        "labeled_none": float(wafers.loc[none, "n_fail"].sum()),
        "labeled_patterned": float(wafers.loc[lab & ~none, "n_fail"].sum()),
    }
    return {k: {"dies_lost": v, "share": v / total} for k, v in parts.items()}


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the
    two empirical distribution functions. 0 means identical distributions,
    1 means no overlap."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def representativeness(wafers: pd.DataFrame) -> pd.DataFrame:
    """Compare wafer yield for labeled and unlabeled wafers, by quantile."""
    rows = []
    groups = {
        "labeled": wafers.loc[wafers["labeled"], "wafer_yield"],
        "unlabeled": wafers.loc[~wafers["labeled"], "wafer_yield"],
    }
    for name, y in groups.items():
        y = y.dropna()
        rows.append({
            "group": name, "n_wafers": len(y), "mean": y.mean(),
            "p10": y.quantile(0.10), "p25": y.quantile(0.25), "median": y.median(),
            "p75": y.quantile(0.75), "p90": y.quantile(0.90),
            "share_below_80pct": (y < 0.80).mean(),
        })
    return pd.DataFrame(rows)


def pattern_mix_by_flag(wafers: pd.DataFrame, lots_spc: pd.DataFrame,
                        flag: str = "ooc_high_laney") -> pd.DataFrame:
    """Share of each label among labeled wafers in flagged lots versus other
    lots. Lift = flagged share / other share. Lift above 1 means the pattern
    is over-represented in flagged lots."""
    flags = lots_spc[["lot", flag]]
    d = wafers[wafers["labeled"]].merge(flags, on="lot", how="inner")
    counts = pd.crosstab(d["failure_type"], d[flag])
    counts = counts.reindex(columns=[False, True], fill_value=0)
    counts.columns = ["n_other", "n_flagged"]
    out = counts.reset_index()
    out["share_other"] = out["n_other"] / out["n_other"].sum()
    out["share_flagged"] = out["n_flagged"] / out["n_flagged"].sum()
    out["lift"] = out["share_flagged"] / out["share_other"]
    return out.sort_values("lift", ascending=False).reset_index(drop=True)