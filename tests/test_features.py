import pandas as pd

from fraud_detection.data import split_data
from fraud_detection.features import FEATURES, add_features, build_xy, filter_fraud_prone


def test_balance_errors_are_zero_for_consistent_transfer():
    row = pd.DataFrame(
        {
            "step": [25],
            "type": ["TRANSFER"],
            "amount": [100.0],
            "oldbalanceOrg": [500.0],
            "newbalanceOrig": [400.0],
            "oldbalanceDest": [50.0],
            "newbalanceDest": [150.0],
        }
    )
    out = add_features(row).iloc[0]
    assert out["errorBalanceOrig"] == 0
    assert out["errorBalanceDest"] == 0
    assert out["origEmptied"] == 0
    assert out["destBalancesZero"] == 0
    assert out["isTransfer"] == 1
    assert out["hourOfDay"] == 1


def test_fraud_signature_flags():
    row = pd.DataFrame(
        {
            "step": [1],
            "type": ["CASH_OUT"],
            "amount": [100.0],
            "oldbalanceOrg": [100.0],
            "newbalanceOrig": [0.0],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [0.0],
        }
    )
    out = add_features(row).iloc[0]
    assert out["origEmptied"] == 1
    assert out["destBalancesZero"] == 1
    assert out["errorBalanceDest"] == 100


def test_filter_keeps_only_fraud_prone_types(transactions):
    kept = filter_fraud_prone(transactions)
    assert set(kept["type"].astype(str)) == {"TRANSFER", "CASH_OUT"}
    assert kept["isFraud"].sum() == transactions["isFraud"].sum()


def test_build_xy_does_not_mutate_input(transactions):
    before = transactions.copy()
    X, y = build_xy(transactions)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(transactions)
    pd.testing.assert_frame_equal(transactions, before)


def test_split_is_disjoint_and_stratified(transactions):
    train, val, test = split_data(transactions)
    assert len(train) + len(val) + len(test) == len(transactions)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)
    overall = transactions["isFraud"].mean()
    for part in (train, val, test):
        assert abs(part["isFraud"].mean() - overall) < 0.01
