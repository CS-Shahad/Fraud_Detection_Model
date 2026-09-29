"""Candidate models compared in this project."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def signed_log1p(x):
    """Compress heavy-tailed amounts while keeping the sign of balance errors."""
    return np.sign(x) * np.log1p(np.abs(x))


def get_models(random_state: int = RANDOM_STATE) -> dict[str, ClassifierMixin]:
    """Return fresh, unfitted instances of every model to compare."""
    return {
        "Logistic Regression": make_pipeline(
            FunctionTransformer(signed_log1p),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=random_state,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        ),
    }
