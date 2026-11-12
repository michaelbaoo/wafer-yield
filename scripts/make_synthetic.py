"""Build a small fake LSWMD.pkl with the same structure as the real file.

Used only for testing the pipeline before the real 2 GB file is downloaded.
Numbers from this file mean nothing. Do not report them.

Usage:
    python scripts/make_synthetic.py --out data/synthetic.pkl
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def make_map(rng: np.random.Generator, size: int, fail_rate: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    on_wafer = (yy - c) ** 2 + (xx - c) ** 2 <= c ** 2
    m = np.zeros((size, size), dtype=np.uint8)
    m[on_wafer] = np.where(rng.random(on_wafer.sum()) < fail_rate, 2, 1)
    return m


def build(n_lots: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lot in range(1, n_lots + 1):
        lot_fail = rng.uniform(0.02, 0.25)
        size = int(rng.choice([26, 32, 40]))
        for w in range(1, 26):
            m = make_map(rng, size, lot_fail)
            if rng.random() < 0.3:
                label = np.array([[rng.choice(LABELS)]], dtype=object)
                split = np.array([["Training"]], dtype=object)
            else:
                label = np.array([], dtype=object)
                split = np.array([], dtype=object)
            rows.append({
                "waferMap": m,
                "dieSize": float((m > 0).sum()),
                "lotName": f"lot{lot}",
                "waferIndex": float(w),
                "trianTestLabel": split,
                "failureType": label,
            })
    # One empty wafer to exercise the no-die path.
    rows.append({"waferMap": np.zeros((5, 5), dtype=np.uint8), "dieSize": 0.0,
                 "lotName": "lot_empty", "waferIndex": 1.0,
                 "trianTestLabel": np.array([], dtype=object),
                 "failureType": np.array([], dtype=object)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic.pkl")
    args = parser.parse_args()
    build().to_pickle(args.out)
    print(f"Wrote {args.out}")
