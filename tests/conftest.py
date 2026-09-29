import numpy as np
import pandas as pd
import pytest


def make_transactions(n: int = 4000, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Small synthetic frame with the PaySim schema and a learnable fraud pattern."""
    rng = np.random.default_rng(seed)
    types = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], size=n)
    amount = rng.lognormal(10, 1.5, size=n).round(2)
    old_orig = amount + rng.lognormal(9, 2, size=n)
    new_orig = old_orig - amount
    old_dest = rng.lognormal(10, 2, size=n)
    new_dest = old_dest + amount

    prone = np.isin(types, ["TRANSFER", "CASH_OUT"])
    fraud = prone & (rng.random(n) < fraud_rate * 2.5)
    # Fraud drains the sender and leaves the receiver's balances unchanged.
    old_orig[fraud] = amount[fraud]
    new_orig[fraud] = 0
    new_dest[fraud] = old_dest[fraud]

    return pd.DataFrame(
        {
            "step": rng.integers(1, 744, size=n),
            "type": pd.Categorical(types),
            "amount": amount,
            "nameOrig": [f"C{i}" for i in range(n)],
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": new_orig,
            "nameDest": [f"C{i + n}" for i in range(n)],
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            "isFraud": fraud.astype("int8"),
            "isFlaggedFraud": (fraud & (amount > 200_000)).astype("int8"),
        }
    )


@pytest.fixture
def transactions() -> pd.DataFrame:
    return make_transactions()
