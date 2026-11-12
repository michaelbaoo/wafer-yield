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


def make_map(rng: np.random.Generator, size: int, fail_rate: float,
             pattern: str = "none") -> np.ndarray:
    """Background random fails at fail_rate, plus a rough version of the
    named defect pattern. Shapes are simplified, not realistic."""
    yy, xx = np.mgrid[:size, :size]
    c = (size - 1) / 2
    dy, dx = yy - c, xx - c
    r = np.sqrt(dy ** 2 + dx ** 2) / c
    theta = np.arctan2(dy, dx)
    on_wafer = r <= 1
    fail = rng.random((size, size)) < fail_rate

    a0 = rng.uniform(-np.pi, np.pi)
    if pattern == "Center":
        fail |= (r < 0.3) & (rng.random(r.shape) < 0.8)
    elif pattern == "Donut":
        fail |= (r > 0.35) & (r < 0.6) & (rng.random(r.shape) < 0.8)
    elif pattern == "Edge-Ring":
        fail |= (r > 0.85) & (rng.random(r.shape) < 0.8)
    elif pattern == "Edge-Loc":
        fail |= (r > 0.75) & (np.abs(np.angle(np.exp(1j * (theta - a0)))) < 0.6) & (rng.random(r.shape) < 0.8)
    elif pattern == "Loc":
        cy, cx = rng.uniform(-0.4, 0.4, 2) * c + c
        fail |= ((yy - cy) ** 2 + (xx - cx) ** 2 < (0.15 * size) ** 2) & (rng.random(r.shape) < 0.8)
    elif pattern == "Scratch":
        t = np.linspace(-0.7, 0.7, 4 * size)
        ys = np.clip(np.round(c + t * c * np.sin(a0)), 0, size - 1).astype(int)
        xs = np.clip(np.round(c + t * c * np.cos(a0)), 0, size - 1).astype(int)
        fail[ys, xs] = True
    elif pattern == "Random":
        fail |= rng.random(r.shape) < 0.4
    elif pattern == "Near-full":
        fail |= rng.random(r.shape) < 0.9

    m = np.zeros((size, size), dtype=np.uint8)
    m[on_wafer] = np.where(fail[on_wafer], 2, 1)
    return m


def build(n_lots: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lot in range(1, n_lots + 1):
        lot_fail = rng.uniform(0.02, 0.10)
        size = int(rng.choice([26, 32, 40]))
        for w in range(1, 26):
            pattern = str(rng.choice(LABELS))
            m = make_map(rng, size, lot_fail, pattern)
            if rng.random() < 0.3:
                label = np.array([[pattern]], dtype=object)
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