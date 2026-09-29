"""Loading the PaySim CSV and splitting it into train / validation / test sets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, KAGGLE_DATASET, RANDOM_STATE, RAW_FILENAME, TARGET, TEST_SIZE, VAL_SIZE

# Compact dtypes cut memory use for the 6.3M-row file by roughly half.
DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def find_data_file(path: str | Path | None = None, download: bool = True) -> Path:
    """Return the path to the PaySim CSV.

    Looks at ``path`` first, then ``data/``. If neither exists and ``download`` is
    true, fetches the dataset from Kaggle with ``kagglehub``.
    """
    candidates = [Path(path)] if path else []
    candidates.append(DATA_DIR / RAW_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if not download:
        raise FileNotFoundError(
            f"Dataset not found. Download it from https://www.kaggle.com/datasets/{KAGGLE_DATASET} "
            f"and place {RAW_FILENAME} in {DATA_DIR}."
        )

    import kagglehub

    downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    return next(downloaded.rglob("*.csv"))


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read the PaySim CSV with memory-friendly dtypes."""
    return pd.read_csv(path, dtype=DTYPES)


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
