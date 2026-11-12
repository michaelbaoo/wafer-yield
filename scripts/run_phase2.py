"""Phase 2: SPC charts on lot fail rate.

Reads outputs/lots.csv from phase 1, so the 2 GB pickle is not reloaded.

Usage (from the project root):
    python scripts/run_phase2.py
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
from wafer_yield.spc import order_lots, p_charts, spc_summary  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

BLUE, RED, GRAY = "#3b6ea5", "#c0392b", "#888888"


def plot_chart(df: pd.DataFrame, chart: str, title: str, path: Path, window: slice | None = None) -> None:
    d = df.iloc[window] if window is not None else df
    ooc = d[f"ooc_high_{chart}"] | d[f"ooc_low_{chart}"]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(d.loc[~ooc, "seq"], d.loc[~ooc, "fail_rate"], s=3, color=BLUE, label="In control")
    ax.scatter(d.loc[ooc, "seq"], d.loc[ooc, "fail_rate"], s=6, color=RED, label="Out of control")
    # Limits differ per lot because each lot has a different die count.
    ax.step(d["seq"], d[f"ucl_{chart}"], where="mid", color=GRAY, lw=0.6, label="Control limits (per lot)")
    ax.step(d["seq"], d[f"lcl_{chart}"], where="mid", color=GRAY, lw=0.6)
    ax.axhline(d["center"].iloc[0], color="black", lw=0.8, label="Center line")
    ax.set_xlabel("Lot sequence (ordered by lot number)")
    ax.set_ylabel("Lot fail rate")
    ax.set_title(title)
    ax.legend(loc="upper right", markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lots", default="outputs/lots.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lots = pd.read_csv(args.lots)
    df, params = p_charts(order_lots(lots))
    stats = spc_summary(df, params)

    df.to_csv(out / "lots_spc.csv", index=False)
    cols = ["seq", "lot", "n_wafers", "n_die", "n_fail", "fail_rate", "ucl_laney", "excess_fails"]
    (df[df["ooc_high_laney"]]
        .sort_values("excess_fails", ascending=False)[cols]
        .to_csv(out / "ooc_lots_laney.csv", index=False))
    (out / "spc_summary.json").write_text(json.dumps(stats, indent=2))

    plot_chart(df, "std", "Standard p-chart: lot fail rate", out / "p_chart_standard.png")
    plot_chart(df, "laney", f"Laney p' chart: lot fail rate (sigma_z = {params['sigma_z']:.1f})",
               out / "p_chart_laney.png")
    # The full series is too dense to read lot by lot, so also plot the first 500.
    plot_chart(df, "laney", "Laney p' chart, first 500 lots", out / "p_chart_laney_first500.png",
               window=slice(0, 500))

    print(json.dumps(stats, indent=2))
    print(f"\nWrote SPC tables and charts to {out.resolve()}")


if __name__ == "__main__":
    main()