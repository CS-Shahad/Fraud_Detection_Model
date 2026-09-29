"""Score new transactions with the saved model."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR
from .features import FRAUD_PRONE_TYPES, add_features


class FraudScorer:
    """Loads the model bundle written by ``fraud_detection.train``."""

    def __init__(self, path: str | Path = MODELS_DIR / "fraud_model.joblib"):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.features = bundle["features"]
        self.name = bundle["name"]

    def score(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Return fraud probability and a flag for each transaction.

        Transactions whose type never carries fraud get probability 0.
        """
        proba = np.zeros(len(transactions))
        prone = transactions["type"].isin(FRAUD_PRONE_TYPES).to_numpy()
        if prone.any():
            X = add_features(transactions[prone])[self.features]
            proba[prone] = self.model.predict_proba(X)[:, 1]
        return pd.DataFrame(
            {"fraud_probability": proba, "is_fraud_alert": proba >= self.threshold},
            index=transactions.index,
        )
