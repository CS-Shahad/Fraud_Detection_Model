"""Project-wide constants and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_DATASET = "ealtaf/paysim1"
RAW_FILENAME = "PS_20174392719_1491204439457_log.csv"

TARGET = "isFraud"
RANDOM_STATE = 42

# Share of rows held out for threshold tuning and for the final test.
VAL_SIZE = 0.2
TEST_SIZE = 0.2
