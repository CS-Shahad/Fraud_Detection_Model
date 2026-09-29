"""Loading the PaySim CSV and splitting it into train / validation / test sets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, KAGGLE_DATASET, RANDOM_STATE, RAW_FILENAME, TARGET, TEST_SIZE, VAL_SIZE

# Compact dtypes, and skipping the account-ID columns (unused as features), keep the
# 6.3M-row file under 1 GB in memory.
DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def find_data_file(path: str | Path | None = None, download: bool = True) -> Path:
    """Return the path to the PaySim CSV.

    Looks at ``path`` first, then for a CSV (or the zip Kaggle serves) in ``data/``.
    If none is found and ``download`` is true, fetches the dataset from Kaggle with
    ``kagglehub``.
    """
    if path:
        return Path(path)
    candidates = [DATA_DIR / RAW_FILENAME, *sorted(DATA_DIR.glob("*.csv")), *sorted(DATA_DIR.glob("*.zip"))]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    manual_steps = (
        f"Download the dataset from https://www.kaggle.com/datasets/{KAGGLE_DATASET} and put the "
        f"zip or CSV in {DATA_DIR}, or set KAGGLE_API_TOKEN (or KAGGLE_USERNAME and KAGGLE_KEY) "
        "so it can be downloaded automatically."
    )
    if not download:
        raise FileNotFoundError(f"Dataset not found. {manual_steps}")

    import kagglehub
    import requests

    try:
        downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except requests.RequestException as err:
        raise RuntimeError(f"Could not download the dataset from Kaggle ({err}).\n{manual_steps}") from None
    return next(downloaded.rglob("*.csv"))


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read the PaySim CSV (plain or zipped) with memory-friendly dtypes."""
    return pd.read_csv(path, usecols=list(DTYPES), dtype=DTYPES)


def split_data(
    df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split that keeps the fraud rate equal in each part."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    train, val = train_test_split(
        train_val,
        test_size=val_size / (1 - test_size),
        stratify=train_val[TARGET],
        random_state=random_state,
    )
    return train, val, test
