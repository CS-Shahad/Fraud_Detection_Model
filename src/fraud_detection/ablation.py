"""Where does the improvement over the original notebook come from?

Starts from the original notebook's setup and adds this project's changes one at a
time, scoring every step on the same test set. Also runs leakage checks.

Usage::

    python -m fraud_detection.ablation
    python -m fraud_detection.ablation --models XGBoost LightGBM   # skip the slow Random Forest
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from .config import RANDOM_STATE, REPORTS_DIR, TARGET
from .data import find_data_file, load_transactions, split_data
from .evaluation import best_f1_threshold, classification_metrics
from .features import FEATURES, FRAUD_PRONE_TYPES, add_features
from .models import get_models

# Features and models exactly as in the original notebook.
ORIGINAL_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]


def original_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    }


STABILISED = ("XGBoost", "LightGBM")


def original_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[ORIGINAL_FEATURES[:6]].copy()
    for t in ("CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"):
        X[f"type_{t}"] = (df["type"] == t).astype("int8")
    return X


def engineered_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df)[FEATURES]


def _fit_score(model, make_X, train, val, test, prone_only: bool):
    """Fit on train, return (validation proba, test proba) over the full val/test sets.

    With ``prone_only``, the model only sees TRANSFER / CASH_OUT rows and every other
    row gets probability 0, so all steps are scored on the same test rows.
    """

    def subset(df):
        return df[df["type"].isin(FRAUD_PRONE_TYPES)] if prone_only else df

    fit_rows = subset(train)
    model.fit(make_X(fit_rows), fit_rows[TARGET])
    probas = []
    for part in (val, test):
        rows = subset(part)
        proba = pd.Series(0.0, index=part.index)
        proba[rows.index] = model.predict_proba(make_X(rows))[:, 1]
        probas.append(proba.to_numpy())
    return probas


def run(df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    train, val, test = split_data(df)
    y_val, y_test = val[TARGET].to_numpy(), test[TARGET].to_numpy()

    rows = []

    def record(step, name, val_proba, test_proba, tune):
        threshold = best_f1_threshold(y_val, val_proba) if tune else 0.5
        m = classification_metrics(y_test, test_proba, threshold)
        rows.append({"step": step, "model": name, **m})
        print(
            f"{name:<14} {step:<48} PR-AUC {m['pr_auc']:.4f}  precision {m['precision']:.4f}  "
            f"recall {m['recall']:.4f}"
        )

    for name in model_names:
        val_p, test_p = _fit_score(original_models()[name], original_matrix, train, val, test, False)
        record("1. Original notebook setup", name, val_p, test_p, tune=False)
        record("2. + threshold tuned on validation", name, val_p, test_p, tune=True)

        val_p, test_p = _fit_score(original_models()[name], original_matrix, train, val, test, True)
        record("3. + train only on TRANSFER / CASH_OUT", name, val_p, test_p, tune=True)

        val_p, test_p = _fit_score(original_models()[name], engineered_matrix, train, val, test, True)
        record("4. + engineered features", name, val_p, test_p, tune=True)

        final = get_models()[name]
        val_p, test_p = _fit_score(final, engineered_matrix, train, val, test, True)
        record("5. + final hyperparameters (= final pipeline)", name, val_p, test_p, tune=True)

        # Diagnostic for boosted trees: with ~0.1% fraud, the first trees can push
        # predictions to exactly 0 or 1, after which training stalls. Capping each
        # tree's output (max_delta_step) is XGBoost's documented remedy.
        if name in STABILISED:
            original = original_models()[name].set_params(max_delta_step=1)
            val_p, test_p = _fit_score(original, original_matrix, train, val, test, False)
            record("check: step 1 with max_delta_step=1", name, val_p, test_p, tune=False)
            final.set_params(max_delta_step=1)
            val_p, test_p = _fit_score(final, engineered_matrix, train, val, test, True)
            record("check: step 5 with max_delta_step=1", name, val_p, test_p, tune=True)

    return pd.DataFrame(rows)


def leakage_checks(df: pd.DataFrame) -> list[str]:
    """Checks that would expose a leak in the final pipeline's setup."""
    lines = []
    forbidden = {TARGET, "isFlaggedFraud"}
    lines.append(f"- Target columns among the features: {sorted(forbidden & set(FEATURES)) or 'none'}")

    prone = df[df["type"].isin(FRAUD_PRONE_TYPES)]
    train, val, test = split_data(prone)
    overlap = len(set(train.index) & set(test.index)) + len(set(val.index) & set(test.index))
    lines.append(f"- Rows shared between the test set and train/validation: {overlap}")

    # With shuffled labels there is nothing real to learn. If the pipeline leaked the
    # answer somehow, PR-AUC would stay high; without a leak it falls to the fraud rate.
    rng = np.random.default_rng(RANDOM_STATE)
    model = get_models()["XGBoost"]
    model.fit(engineered_matrix(train), rng.permutation(train[TARGET].to_numpy()))
    shuffled = average_precision_score(test[TARGET], model.predict_proba(engineered_matrix(test))[:, 1])
    lines.append(
        f"- XGBoost trained on shuffled labels: test PR-AUC {shuffled:.4f} "
        f"(chance level = fraud rate {test[TARGET].mean():.4f})"
    )
    return lines


def to_markdown(results: pd.DataFrame, checks: list[str]) -> str:
    table = results[["model", "step", "pr_auc", "precision", "recall", "f1", "fp", "fn"]].rename(
        columns={"pr_auc": "PR-AUC", "fp": "false alarms", "fn": "missed fraud"}
    )
    return (
        "# Where the improvement comes from\n\n"
        "Each step adds one change to the original notebook's setup. All steps are scored on the same "
        "stratified 20% test set of all 6.3M transactions (rows outside TRANSFER / CASH_OUT count as "
        "predicted legitimate), so the numbers differ slightly from the main results.\n\n"
        + table.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Leakage checks\n\n"
        + "\n".join(checks)
        + "\n"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", help="Path to the PaySim CSV")
    parser.add_argument(
        "--models", nargs="+", default=list(original_models()), choices=list(original_models())
    )
    args = parser.parse_args(argv)

    df = load_transactions(find_data_file(args.data))
    results = run(df, args.models)
    print("\nLeakage checks")
    checks = leakage_checks(df)
    print("\n".join(checks))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "ablation.md").write_text(to_markdown(results, checks))
    print(f"\nWritten to {REPORTS_DIR / 'ablation.md'}")


if __name__ == "__main__":
    main()
