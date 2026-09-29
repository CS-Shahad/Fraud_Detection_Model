"""Feature engineering for PaySim transactions."""

from __future__ import annotations

import pandas as pd

from .config import TARGET

# Every fraudulent transaction in PaySim is a TRANSFER or a CASH_OUT, so the other
# types can be marked legitimate by rule and left out of model training.
FRAUD_PRONE_TYPES = ("TRANSFER", "CASH_OUT")

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
    "origEmptied",
    "destBalancesZero",
    "isTransfer",
    "hourOfDay",
]


def filter_fraud_prone(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only transaction types in which fraud occurs."""
    return df[df["type"].isin(FRAUD_PRONE_TYPES)]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered feature columns added.

    - ``errorBalanceOrig`` / ``errorBalanceDest``: how far the recorded balances are
      from what the amount implies. Legitimate transfers mostly balance out;
      fraudulent ones often don't.
    - ``origEmptied``: the sender's account was drained to zero.
    - ``destBalancesZero``: a non-zero amount arrived, yet the receiver shows zero
      balance before and after.
    - ``hourOfDay``: ``step`` counts hours from the start of the simulation.
    """
    out = df.copy()
    out["errorBalanceOrig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["errorBalanceDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    out["origEmptied"] = ((out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)).astype("int8")
    out["destBalancesZero"] = (
        (out["oldbalanceDest"] == 0) & (out["newbalanceDest"] == 0) & (out["amount"] > 0)
    ).astype("int8")
    out["isTransfer"] = (out["type"] == "TRANSFER").astype("int8")
    out["hourOfDay"] = (out["step"] % 24).astype("int8")
    return out


def build_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer features and return the model matrix ``X`` and target ``y``."""
    featured = add_features(df)
    return featured[FEATURES], featured[TARGET]
