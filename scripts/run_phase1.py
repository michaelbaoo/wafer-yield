"""Phase 1: load WM-811K, compute wafer and lot yield, write tables and charts.

Usage (from the project root):
    python scripts/run_phase1.py --data data/LSWMD.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_yield.load import load_wafers  # noqa: E402
from wafer_yield.metrics import label_distribution, lot_yield, summary  # noqa: E402

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/LSWMD.pkl")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading {args.data} ...")
    wafers, _maps = load_wafers(args.data)
    print(f"Loaded {len(wafers):,} wafers in {time.time() - t0:.0f}s")

    lots = lot_yield(wafers)
    labels = label_distribution(wafers)
    stats = summary(wafers, lots)

    wafers.to_csv(out / "wafers.csv", index=False)
    lots.to_csv(out / "lots.csv", index=False)
    labels.to_csv(out / "label_distribution.csv", index=False)
    (out / "summary.json").write_text(json.dumps(stats, indent=2))

    # Chart 1: distribution of lot yield
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lots["lot_yield"], bins=50, color="#3b6ea5")
    ax.set_xlabel("Lot yield")
    ax.set_ylabel("Number of lots")
    ax.set_title("Lot yield distribution")
    fig.tight_layout()
    fig.savefig(out / "lot_yield_hist.png", dpi=150)
    plt.close(fig)

    # Chart 2: defect label counts among labeled wafers
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels["failure_type"], labels["n_wafers"], color="#3b6ea5")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Labeled wafers (log scale)")
    ax.set_title("Defect pattern counts")
    fig.tight_layout()
    fig.savefig(out / "label_counts.png", dpi=150)
    plt.close(fig)

    print(json.dumps(stats, indent=2))
    print(f"\nWrote tables and charts to {out.resolve()}")


if __name__ == "__main__":
    main()
