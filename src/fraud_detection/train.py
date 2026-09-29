"""End-to-end training pipeline.

Usage::

    python -m fraud_detection.train                     # finds or downloads the data
    python -m fraud_detection.train --data path/to.csv
    python -m fraud_detection.train --sample 0.1        # quick run on 10% of rows
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import plots
from .config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from .data import find_data_file, load_transactions, split_data
from .evaluation import best_f1_threshold, classification_metrics, comparison_table
from .features import FEATURES, build_xy, filter_fraud_prone
from .models import get_models


@dataclass
class TrainingResult:
    models: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    val_pr_auc: dict = field(default_factory=dict)
    test_probas: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    fit_seconds: dict = field(default_factory=dict)
    y_test: pd.Series | None = None
    test_amount: pd.Series | None = None
    baseline_metrics: dict | None = None

    @property
    def best_model_name(self) -> str:
        # Chosen on validation data so the test set stays an unbiased final check.
        return max(self.val_pr_auc, key=self.val_pr_auc.get)


def train_and_evaluate(df: pd.DataFrame, models: dict | None = None, verbose: bool = True) -> TrainingResult:
    """Fit every model on the training split, tune its threshold on validation, score on test."""
    df = filter_fraud_prone(df)
    train, val, test = split_data(df)
    X_train, y_train = build_xy(train)
    X_val, y_val = build_xy(val)
    X_test, y_test = build_xy(test)

    result = TrainingResult(y_test=y_test, test_amount=test["amount"])
    # The simulator's own rule-based flag is the baseline every model has to beat.
    result.baseline_metrics = classification_metrics(
        y_test, test["isFlaggedFraud"], threshold=0.5, amount=test["amount"]
    )

    for name, model in (models or get_models()).items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        result.fit_seconds[name] = time.perf_counter() - start

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = best_f1_threshold(y_val, val_proba)
        result.val_pr_auc[name] = average_precision_score(y_val, val_proba)
        proba = model.predict_proba(X_test)[:, 1]

        result.models[name] = model
        result.thresholds[name] = threshold
        result.test_probas[name] = proba
        result.test_metrics[name] = classification_metrics(y_test, proba, threshold, amount=test["amount"])
        if verbose:
            m = result.test_metrics[name]
            print(
                f"{name:<20} PR-AUC {m['pr_auc']:.4f}  precision {m['precision']:.4f}  "
                f"recall {m['recall']:.4f}  ({result.fit_seconds[name]:.0f}s)"
            )
    return result


def feature_importances(model) -> pd.Series | None:
    """Normalised importances for tree models; ``None`` for models without them."""
    if not hasattr(model, "feature_importances_"):
        return None
    values = np.asarray(model.feature_importances_, dtype=float)
    return pd.Series(values / values.sum(), index=FEATURES)


def best_tree_model_name(result: TrainingResult) -> str | None:
    """Best model by validation PR-AUC among those that expose feature importances."""
    ranked = sorted(result.val_pr_auc, key=result.val_pr_auc.get, reverse=True)
    return next((n for n in ranked if feature_importances(result.models[n]) is not None), None)


def save_outputs(result: TrainingResult) -> None:
    """Write metrics, figures and the best model bundle to disk."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best = result.best_model_name

    all_metrics = {"Rule baseline (isFlaggedFraud)": result.baseline_metrics, **result.test_metrics}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps({"best_model": best, "test": all_metrics}, indent=2))
    comparison_table(all_metrics).to_markdown(
        REPORTS_DIR / "model_comparison.md", floatfmt=("",) + (".4f",) * 6 + (".0f",) * 2
    )

    plots.apply_style()
    b = result.baseline_metrics
    plots.precision_recall_curves(
        result.y_test,
        result.test_probas,
        baseline=(b["recall"], b["precision"], "isFlaggedFraud rule"),
        path=FIGURES_DIR / "precision_recall_curves.png",
    )
    plots.confusion_matrix_plot(
        result.test_metrics[best], f"{best} on the test set", path=FIGURES_DIR / "confusion_matrix.png"
    )
    name = best_tree_model_name(result)
    if name is not None:
        plots.feature_importance(
            feature_importances(result.models[name]),
            f"What drives the {name} model",
            path=FIGURES_DIR / "feature_importance.png",
        )

    joblib.dump(
        {
            "model": result.models[best],
            "threshold": result.thresholds[best],
            "features": FEATURES,
            "name": best,
        },
        MODELS_DIR / "fraud_model.joblib",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", help="Path to the PaySim CSV (default: data/ or download from Kaggle)")
    parser.add_argument("--sample", type=float, help="Fraction of rows to use, for a quick run")
    args = parser.parse_args(argv)

    path = find_data_file(args.data)
    print(f"Loading {path}")
    df = load_transactions(path)
    if args.sample:
        df = df.groupby(TARGET, group_keys=False).sample(frac=args.sample, random_state=RANDOM_STATE)
    print(f"{len(df):,} transactions, {df[TARGET].sum():,} fraudulent")

    result = train_and_evaluate(df)
    save_outputs(result)
    print(f"\nBest model: {result.best_model_name}. Outputs written to {REPORTS_DIR} and {MODELS_DIR}.")


if __name__ == "__main__":
    main()
