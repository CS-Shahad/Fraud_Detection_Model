# Fraud Detection in Mobile-Money Transactions

[![tests](https://github.com/CS-Shahad/Fraud_Detection_Model/actions/workflows/tests.yml/badge.svg)](https://github.com/CS-Shahad/Fraud_Detection_Model/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CS-Shahad/Fraud_Detection_Model/blob/main/notebooks/fraud_detection.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A machine-learning pipeline that detects fraudulent transactions among **6.3 million** mobile-money
transactions, where only **0.13%** are fraud. It compares Logistic Regression, Random Forest, XGBoost and
LightGBM against the rule-based system already in the data.

## Highlights

- **Evaluation built for imbalanced data.** Models are judged on PR-AUC, precision, recall and the share of
  stolen money recovered, not on accuracy, which is 99.87% even for a model that never flags anything.
- **No test-set leakage.** A stratified 60/20/20 train/validation/test split. Decision thresholds and the
  final model are chosen on validation, and the test set is used once.
- **Domain-driven features.** Fraud in this data shows up as account balances that don't add up after a
  transaction. Engineered balance-error features capture that directly.
- **Reusable code, not only a notebook.** A small Python package with a command-line training script, a
  scorer for new transactions, unit tests and CI.

## Results

<!-- RESULTS -->
The results table and charts are produced by `python -m fraud_detection.train` and saved to
[`reports/`](reports/).
<!-- /RESULTS -->

## Approach

**1. Explore.** All fraud occurs in `TRANSFER` and `CASH_OUT` transactions, which fits the fraud scenario
the simulator models: take over an account, transfer the funds to a mule account, and cash out. The other
three types are labelled legitimate by rule, so over half of the rows never reach the model.

**2. Engineer features.**

| Feature | Idea |
|---|---|
| `errorBalanceOrig`, `errorBalanceDest` | Gap between the recorded balances and what the amount implies |
| `origEmptied` | The sender's account was drained to exactly zero |
| `destBalancesZero` | Money arrived, yet the receiver shows a zero balance before and after |
| `isTransfer`, `hourOfDay` | Transaction type and time of day |

**3. Train and tune.** Each model is fit on the training set. Its decision threshold is then set on the
validation set to maximise F1. Class imbalance is handled through the threshold rather than
oversampling, which keeps the predicted probabilities meaningful.

**4. Evaluate once** on the untouched test set, against the `isFlaggedFraud` rule as the baseline.

## Project structure

```
├── notebooks/
│   └── fraud_detection.ipynb   # the full analysis, with charts and commentary
├── src/fraud_detection/
│   ├── data.py                 # loading (with Kaggle download) and stratified splitting
│   ├── features.py             # feature engineering
│   ├── models.py               # the candidate models
│   ├── evaluation.py           # PR-AUC, threshold tuning, fraud amount caught
│   ├── plots.py                # charts
│   ├── train.py                # end-to-end training CLI
│   └── predict.py              # FraudScorer: score new transactions
├── tests/                      # unit and pipeline tests (pytest)
├── reports/                    # metrics and figures from the latest run
└── data/                       # dataset goes here (not committed)
```

## Run it

**In the browser:** click *Open in Colab* above and run all cells. The dataset downloads automatically.

**In GitHub Codespaces:** *Code → Codespaces → Create codespace*. Dependencies install automatically, then
run `python -m fraud_detection.train` in the terminal.

**Locally:**

```bash
git clone https://github.com/CS-Shahad/Fraud_Detection_Model.git
cd Fraud_Detection_Model
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m fraud_detection.train              # full run: downloads data, trains, writes reports/ and models/
python -m fraud_detection.train --sample 0.1 # quick run on 10% of the data
pytest                                       # tests
```

**Score new transactions:**

```python
import pandas as pd
from fraud_detection.predict import FraudScorer

scorer = FraudScorer()  # loads models/fraud_model.joblib
scorer.score(pd.read_csv("new_transactions.csv"))  # -> fraud_probability, is_fraud_alert
```

## Limitations and next steps

- **Simulated data.** PaySim imitates real mobile-money logs, but the balance-error signal is probably
  cleaner here than in real transactions, so real-world performance would be lower.
- **Timing.** The features use balances *after* the transaction, so the model detects fraud just after it
  happens rather than blocking it before authorisation.
- **Next:** validate on a time-based split, choose the threshold from a business cost model, add
  per-account behaviour features, explain predictions with SHAP, and serve the scorer through an API.

## Dataset

[PaySim on Kaggle](https://www.kaggle.com/datasets/ealtaf/paysim1). E. A. Lopez-Rojas, A. Elmir and
S. Axelsson, *PaySim: A financial mobile money simulator for fraud detection*, EMSS 2016.
