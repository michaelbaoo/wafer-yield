"""Phase 4, step 2: train and evaluate the classifier, then label every
unlabeled wafer and rerun the Pareto on all yield loss.

Needs outputs from build_features.py.

Usage (from the project root):
    python scripts/run_phase4.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wafer_yield.classify import (  # noqa: E402
    evaluate, make_model, pareto_with_predictions, split_by_lot,
)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]

CONF_THRESHOLD = 0.8


def plot_confusion(cm: pd.DataFrame, path: Path) -> None:
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)  # row = true class, sums to 1
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(cm)), cm.index)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = norm.iat[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix, test lots (row-normalized recall)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    out = Path(args.out)

    X = np.load(out / "features.npy")
    index = pd.read_csv(out / "feature_index.csv")
    labeled = index["labeled"].to_numpy()

    Xl = X[labeled]
    yl = index.loc[labeled, "failure_type"].to_numpy()
    lots = index.loc[labeled, "lot"].to_numpy()
    classes = sorted(set(yl))

    tr, te = split_by_lot(lots)
    assert not set(lots[tr]) & set(lots[te]), "lot leakage between train and test"
    print(f"Train {len(tr):,} wafers / test {len(te):,} wafers, split by lot")

    t0 = time.time()
    model = make_model().fit(Xl[tr], yl[tr])
    print(f"Trained in {time.time() - t0:.0f}s ({model.n_iter_} boosting rounds)")

    report, cm, macro_recall = evaluate(yl[te], model.predict(Xl[te]), classes)
    report.to_csv(out / "classifier_report.csv")
    cm.to_csv(out / "confusion_matrix.csv")
    plot_confusion(cm, out / "confusion_matrix.png")

    # Refit on all labeled wafers before labeling the unlabeled ones.
    final = make_model().fit(Xl, yl)
    joblib.dump(final, out / "classifier.joblib")

    proba = final.predict_proba(X)
    pred = final.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    unl = ~labeled
    pd.DataFrame({"lot": index["lot"], "wafer_index": index["wafer_index"],
                  "labeled": labeled, "true_label": index["failure_type"],
                  "pred_label": pred, "confidence": conf}).to_csv(out / "predictions.csv", index=False)

    pareto_all = pareto_with_predictions(index, pred, conf, 0.0)
    pareto_conf = pareto_with_predictions(index, pred, conf, CONF_THRESHOLD)
    pareto_all.to_csv(out / "pareto_all_wafers.csv", index=False)
    pareto_conf.to_csv(out / f"pareto_all_wafers_conf{int(CONF_THRESHOLD * 100)}.csv", index=False)

    summary = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "macro_recall_test": macro_recall,
        "macro_recall_test_excluding_none": float(report.drop(index="none")["recall"].mean()),
        "accuracy_test": float((model.predict(Xl[te]) == yl[te]).mean()),
        "n_unlabeled": int(unl.sum()),
        "share_unlabeled_confident": float((conf[unl] >= CONF_THRESHOLD).mean()),
        "conf_threshold": CONF_THRESHOLD,
    }
    (out / "phase4_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 160)
    print("\nPer-class results on held-out lots:")
    print(report.round(3).to_string())
    print("\nPareto over ALL wafers (true labels + predictions):")
    print(pareto_all.round(4).to_string(index=False))
    print(f"\nSame, counting predictions below {CONF_THRESHOLD} confidence as 'uncertain':")
    print(pareto_conf.round(4).to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()