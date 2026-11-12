"""Phase 4, step 1: compute engineered features for every wafer.

Loads LSWMD.pkl once (about 30 s) and writes:
    outputs/features.npy        (n_wafers x 85 float32, about 275 MB)
    outputs/feature_index.csv   lot, wafer_index, label, die counts, same row order
    outputs/feature_names.json

This is the slow step (minutes for 811k wafers). Phase 4 training reuses the
saved features, so this runs once.

Usage (from the project root):
    python scripts/build_features.py --data data/LSWMD.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_yield.features import build_matrix, feature_names  # noqa: E402
from wafer_yield.load import load_wafers  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/LSWMD.pkl")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    wafers, maps = load_wafers(args.data)
    print(f"Loaded {len(wafers):,} wafers in {time.time() - t0:.0f}s")

    t1 = time.time()
    X = build_matrix(maps.to_numpy())
    print(f"Built features in {time.time() - t1:.0f}s")

    np.save(out / "features.npy", X)
    wafers[["lot", "wafer_index", "failure_type", "labeled", "n_die", "n_fail"]].to_csv(
        out / "feature_index.csv", index=False)
    (out / "feature_names.json").write_text(json.dumps(feature_names()))
    print(f"Wrote features {X.shape} to {out.resolve()}")


if __name__ == "__main__":
    main()