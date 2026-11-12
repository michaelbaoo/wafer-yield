"""Phase 3: defect Pareto, labeled-set check, and pattern mix in flagged lots.

Reads outputs/wafers.csv (phase 1) and outputs/lots_spc.csv (phase 2).

Usage (from the project root):
    python scripts/run_phase3.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_yield.pareto import (  # noqa: E402
    ks_statistic, labeled_split, pattern_mix_by_flag, pattern_pareto, representativeness,
)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
BLUE, ORANGE, GRAY = "#3b6ea5", "#d9822b", "#888888"


def plot_pareto(p: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: classic Pareto by dies lost, with cumulative share line.
    ax1.bar(p["failure_type"], p["dies_lost"] / 1e6, color=BLUE)
    ax1.set_ylabel("Dies lost (millions)")
    ax1.set_title("Patterned dies lost, ranked")
    ax1.tick_params(axis="x", rotation=35)
    ax2 = ax1.twinx()
    ax2.plot(p["failure_type"], p["cum_share_dies_lost"] * 100, color=ORANGE, marker="o")
    ax2.axhline(80, color=GRAY, lw=0.8, ls="--")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative share (%)")

    # Right: dies lost per wafer, same order. Shows severity, not frequency.
    ax3.bar(p["failure_type"], p["dies_lost_per_wafer"], color=BLUE)
    ax3.set_ylabel("Dies lost per affected wafer")
    ax3.set_title("Severity per occurrence")
    ax3.tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wafers", default="outputs/wafers.csv")
    parser.add_argument("--lots-spc", default="outputs/lots_spc.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wafers = pd.read_csv(args.wafers, usecols=["lot", "failure_type", "labeled",
                                               "n_fail", "n_die", "wafer_yield"])
    lots_spc = pd.read_csv(args.lots_spc, usecols=["lot", "ooc_high_laney"])

    pareto = pattern_pareto(wafers)
    split = labeled_split(wafers)
    rep = representativeness(wafers)
    ks = ks_statistic(
        wafers.loc[wafers["labeled"], "wafer_yield"].dropna().to_numpy(),
        wafers.loc[~wafers["labeled"], "wafer_yield"].dropna().to_numpy(),
    )
    mix = pattern_mix_by_flag(wafers, lots_spc)

    pareto.to_csv(out / "pareto_patterns.csv", index=False)
    rep.to_csv(out / "labeled_vs_unlabeled.csv", index=False)
    mix.to_csv(out / "pattern_mix_flagged_lots.csv", index=False)
    plot_pareto(pareto, out / "pareto.png")

    lots_flagged = set(lots_spc.loc[lots_spc["ooc_high_laney"], "lot"])
    in_flagged = wafers["lot"].isin(lots_flagged)
    summary = {
        "dies_lost_split": split,
        "ks_labeled_vs_unlabeled_yield": ks,
        "patterns_to_reach_80pct_of_patterned_loss":
            int((pareto["cum_share_dies_lost"] < 0.80).sum() + 1),
        "labeled_wafers_in_flagged_lots": int((wafers["labeled"] & in_flagged).sum()),
        "share_of_flagged_lot_wafers_labeled": float(wafers.loc[in_flagged, "labeled"].mean()),
    }
    (out / "phase3_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    print("Pareto (patterned wafers only):")
    print(pareto[["failure_type", "n_wafers", "dies_lost", "dies_lost_per_wafer",
                  "share_dies_lost", "cum_share_dies_lost", "rank_count"]].round(3).to_string(index=False))
    print("\nLabeled vs unlabeled wafer yield:")
    print(rep.round(4).to_string(index=False))
    print("\nPattern mix in Laney-flagged lots vs other lots:")
    print(mix.round(3).to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote phase 3 outputs to {out.resolve()}")


if __name__ == "__main__":
    main()