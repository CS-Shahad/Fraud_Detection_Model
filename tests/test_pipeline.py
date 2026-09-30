import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_detection import train as train_module
from fraud_detection.evaluation import best_f1_threshold, classification_metrics
from fraud_detection.models import get_models
from fraud_detection.predict import FraudScorer


def test_best_f1_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    threshold = best_f1_threshold(y, proba)
    assert 0.3 < threshold <= 0.8
    assert classification_metrics(y, proba, threshold)["f1"] == 1.0


def test_fraud_amount_caught():
    y = np.array([0, 1, 1])
    proba = np.array([0.0, 0.9, 0.1])
    amount = np.array([10.0, 300.0, 100.0])
    assert classification_metrics(y, proba, 0.5, amount)["fraud_amount_caught"] == pytest.approx(0.75)


def test_all_models_train_and_beat_chance(transactions):
    result = train_module.train_and_evaluate(transactions, models=get_models(), verbose=False)
    assert set(result.test_metrics) == set(get_models())
    for metrics in result.test_metrics.values():
        assert metrics["roc_auc"] > 0.9


def test_saved_model_scores_new_transactions(transactions, tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(train_module, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")

    result = train_module.train_and_evaluate(
        transactions, models={"LR": LogisticRegression(max_iter=1000)}, verbose=False
    )
    train_module.save_outputs(result)

    assert (tmp_path / "reports" / "metrics.json").is_file()
    assert (tmp_path / "reports" / "figures" / "precision_recall_curves.png").is_file()

    scores = FraudScorer(tmp_path / "models" / "fraud_model.joblib").score(transactions)
    assert len(scores) == len(transactions)
    not_prone = ~transactions["type"].isin(["TRANSFER", "CASH_OUT"])
    assert (scores.loc[not_prone, "fraud_probability"] == 0).all()


def test_ablation_scores_every_step_on_the_same_test_set(transactions):
    from fraud_detection.ablation import leakage_checks, run

    results = run(transactions, ["XGBoost"])
    assert len(results) == 7
    # Same test rows at every step, so the number of real frauds never changes.
    assert (results["tp"] + results["fn"]).nunique() == 1

    checks = leakage_checks(transactions)
    assert "none" in checks[0]
    assert checks[1].endswith(": 0")
