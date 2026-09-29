"""Metrics suited to heavily imbalanced fraud data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(y_true, proba) -> float:
    """Return the probability cut-off that maximises F1 on the given data.

    Tune this on a validation set, never on the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The last precision/recall pair has no threshold attached.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[np.argmax(f1)])


def classification_metrics(y_true, proba, threshold: float, amount=None) -> dict:
    """Threshold-free and thresholded metrics for the fraud (positive) class.

    If ``amount`` is given, also reports the share of fraudulent money the model
    catches, which is closer to what the business cares about than a row count.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if amount is not None:
        amount = np.asarray(amount)
        fraud_amount = amount[y_true == 1].sum()
        caught = amount[(y_true == 1) & (y_pred == 1)].sum()
        metrics["fraud_amount_caught"] = caught / fraud_amount if fraud_amount else float("nan")
    return metrics


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """One row per model, sorted by PR-AUC."""
    columns = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fraud_amount_caught", "fp", "fn"]
    table = pd.DataFrame(results).T
    table = table[[c for c in columns if c in table.columns]].astype(float)
    table[["fp", "fn"]] = table[["fp", "fn"]].astype(int)
    return table.sort_values("pr_auc", ascending=False)
