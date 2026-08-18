import pandas as pd

from src.feature_engineering import create_features


def test_feature_engineering_creates_log_amount():
    data = pd.DataFrame({
        "transaction_id": ["TXN_001"],
        "customer_id": ["CUST_001"],
        "amount": [10000],
        "transaction_hour": [12],
        "transaction_frequency": [2],
        "location_changed": [0],
        "device_changed": [0],
    })

    result = create_features(data)

    assert "log_amount" in result.columns


def test_night_transaction_is_detected():
    data = pd.DataFrame({
        "transaction_id": ["TXN_001"],
        "customer_id": ["CUST_001"],
        "amount": [10000],
        "transaction_hour": [2],
        "transaction_frequency": [2],
        "location_changed": [0],
        "device_changed": [0],
    })

    result = create_features(data)

    assert result.iloc[0]["is_night"] == 1


def test_risk_signal_is_calculated():
    data = pd.DataFrame({
        "transaction_id": ["TXN_001"],
        "customer_id": ["CUST_001"],
        "amount": [10000],
        "transaction_hour": [2],
        "transaction_frequency": [2],
        "location_changed": [1],
        "device_changed": [1],
    })

    result = create_features(data)

    assert result.iloc[0]["risk_signal"] == 3
