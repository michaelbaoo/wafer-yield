"""Train, evaluate, and apply the defect pattern classifier.

Design choices:
- Train/test split by lot, not by wafer. Wafers in one lot share process
  history, so a random wafer split would leak lot information into the test
  set and overstate accuracy.
- Class weights are balanced, because 'none' is ~85% of labeled wafers. A
  model that always says 'none' would score 85% accuracy and be useless.
- Results are reported as per-class precision and recall and macro recall
  (the plain average of per-class recall), not accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GroupShuffleSplit


def split_by_lot(lots: np.ndarray, test_size: float = 0.2, seed: int = 0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(lots)), groups=lots))
    return train_idx, test_idx


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.1, max_leaf_nodes=31,
        class_weight="balanced", early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=seed,
    )


def evaluate(y_true, y_pred, classes) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    ).T.loc[classes, ["precision", "recall", "f1-score", "support"]]
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes)
    macro_recall = float(recall_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
    return report, cm, macro_recall


def pareto_with_predictions(index: pd.DataFrame, pred: np.ndarray, conf: np.ndarray,
                            min_conf: float = 0.0) -> pd.DataFrame:
    """Dies lost by pattern over all wafers: true labels where they exist,
    predicted labels elsewhere. Unlabeled wafers whose top-class probability
    is below min_conf are counted as 'uncertain'."""
    label = index["failure_type"].astype(object).to_numpy().copy()
    unl = ~index["labeled"].to_numpy()
    use = np.where(conf >= min_conf, pred, "uncertain")
    label[unl] = use[unl]
    d = pd.DataFrame({"pattern": label, "n_fail": index["n_fail"].to_numpy(),
                      "source": np.where(unl, "predicted", "labeled")})
    out = (d.groupby("pattern")
             .agg(n_wafers=("n_fail", "size"), dies_lost=("n_fail", "sum"),
                  n_predicted=("source", lambda s: int((s == "predicted").sum())))
             .sort_values("dies_lost", ascending=False).reset_index())
    out["share_all_dies_lost"] = out["dies_lost"] / out["dies_lost"].sum()
    return out