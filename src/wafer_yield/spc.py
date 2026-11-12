"""Statistical process control on lot fail rate.

Two charts are computed on the same data:

1. Standard p-chart. Each lot's fail rate p_i = fails / dies is compared with
   limits  p_bar +/- 3 * sqrt(p_bar * (1 - p_bar) / n_i).
   This assumes the only variation between lots is binomial sampling noise.
   With ~30,000 dies per lot that noise is tiny, so the limits are very
   tight and real lot-to-lot variation pushes most lots outside them.

2. Laney p' chart. Same center line, but each lot is first converted to a
   z-score  z_i = (p_i - p_bar) / sigma_pi, and the spread of those z-scores
   is measured from lot to lot with the average moving range:
       sigma_z = mean(|z_i - z_(i-1)|) / 1.128
   Limits become  p_bar +/- 3 * sigma_pi * sigma_z.
   sigma_z is about 1 when variation is purely binomial, and larger when
   lots differ for real reasons (overdispersion). Laney's chart is the
   standard correction for large-sample attribute data.

Fail rate is used rather than yield because control charts for defectives
track the fraction defective. Fail rate = 1 - yield.

The moving range depends on lot order. WM-811K has no timestamps, so lots
are ordered by the number in their lot name. That order is an assumption.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

D2 = 1.128  # control chart constant for moving ranges of size 2


def lot_number(name: str) -> float:
    """Extract the trailing integer from a lot name like 'lot1234'.
    Returns NaN when there is no trailing number."""
    m = re.search(r"(\d+)\s*$", str(name))
    return float(m.group(1)) if m else np.nan


def order_lots(lots: pd.DataFrame) -> pd.DataFrame:
    """Sort lots by lot number. Lots without a number go last, in their
    original order, and are flagged."""
    out = lots.copy()
    out["lot_num"] = out["lot"].map(lot_number)
    out["lot_num_missing"] = out["lot_num"].isna()
    out = out.sort_values(["lot_num_missing", "lot_num"], kind="stable").reset_index(drop=True)
    out["seq"] = np.arange(1, len(out) + 1)
    return out


def p_charts(lots: pd.DataFrame, k: float = 3.0) -> tuple[pd.DataFrame, dict]:
    """Add standard and Laney p-chart columns to an ordered lot table.

    Needs columns n_die and n_fail. Returns the table and the chart parameters.
    """
    df = lots.copy()
    n = df["n_die"].to_numpy(dtype=float)
    fails = df["n_fail"].to_numpy(dtype=float)

    p = fails / n
    p_bar = fails.sum() / n.sum()
    sigma_p = np.sqrt(p_bar * (1 - p_bar) / n)

    z = (p - p_bar) / sigma_p
    mr = np.abs(np.diff(z))
    sigma_z = float(mr.mean() / D2) if len(mr) else 1.0

    df["fail_rate"] = p
    df["center"] = p_bar
    df["sigma_p"] = sigma_p
    df["z"] = z

    df["ucl_std"] = p_bar + k * sigma_p
    df["lcl_std"] = np.clip(p_bar - k * sigma_p, 0, None)
    df["ucl_laney"] = p_bar + k * sigma_p * sigma_z
    df["lcl_laney"] = np.clip(p_bar - k * sigma_p * sigma_z, 0, None)

    for chart in ("std", "laney"):
        above = p > df[f"ucl_{chart}"].to_numpy()
        below = p < df[f"lcl_{chart}"].to_numpy()
        df[f"ooc_high_{chart}"] = above   # fail rate too high: yield problem
        df[f"ooc_low_{chart}"] = below    # fail rate unusually low: worth knowing why

    # Dies lost beyond what the process average predicts for this lot size.
    df["excess_fails"] = fails - p_bar * n

    params = {"p_bar": float(p_bar), "sigma_z": sigma_z, "k": k}
    return df, params


def spc_summary(df: pd.DataFrame, params: dict) -> dict:
    """Headline numbers for the yield review."""
    total_fails = float(df["n_fail"].sum())
    out = {
        "n_lots": int(len(df)),
        "n_lots_without_number": int(df["lot_num_missing"].sum()) if "lot_num_missing" in df else 0,
        "center_fail_rate": params["p_bar"],
        "center_yield": 1 - params["p_bar"],
        "sigma_z": params["sigma_z"],
        "median_dies_per_lot": float(df["n_die"].median()),
        "median_std_limit_halfwidth": float((df["ucl_std"] - df["center"]).median()),
        "median_laney_limit_halfwidth": float((df["ucl_laney"] - df["center"]).median()),
    }
    for chart in ("std", "laney"):
        high = df[f"ooc_high_{chart}"]
        out[f"ooc_high_{chart}"] = int(high.sum())
        out[f"ooc_high_{chart}_share"] = float(high.mean())
        out[f"ooc_low_{chart}"] = int(df[f"ooc_low_{chart}"].sum())
        out[f"ooc_high_{chart}_share_of_dies_lost"] = float(df.loc[high, "n_fail"].sum() / total_fails)
        out[f"ooc_high_{chart}_excess_fails"] = float(df.loc[high, "excess_fails"].sum())

    # Expected false alarms on the high side if every lot were in control.
    out["expected_ooc_high_if_in_control"] = float(len(df) * 0.00135)

    # Does lot yield depend on lot size? Spearman rank correlation.
    out["spearman_dies_vs_yield"] = float(
        df["n_die"].rank().corr((1 - df["fail_rate"]).rank())
    )
    return out