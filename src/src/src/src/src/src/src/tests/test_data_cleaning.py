import pandas as pd
import pytest

from src.data_cleaning import (
    clean_transactions,
    validate_columns,
)


def test_validate_columns_accepts_valid_data():
    data = pd.DataFrame({
        "transaction_id": ["TXN_001"],
        "customer_id": ["CUST_001"],
        "amount": [1000],
        "transaction_hour": [12],
        "transaction_frequency": [2],
        "location_changed": [0],
        "device_changed": [0],
    })

    validate_columns(data)


def test_clean_transactions_removes_duplicates():
    data = pd.DataFrame({
        "transaction_id": ["TXN_001", "TXN_001"],
        "customer_id": ["CUST_001", "CUST_001"],
        "amount": [1000, 1000],
        "transaction_hour": [12, 12],
        "transaction_frequency": [2, 2],
        "location_changed": [0, 0],
        "device_changed": [0, 0],
    })

    cleaned = clean_transactions(data)

    assert len(cleaned) == 1


def test_invalid_transaction_amount_is_removed():
    data = pd.DataFrame({
        "transaction_id": ["TXN_001", "TXN_002"],
        "customer_id": ["CUST_001", "CUST_002"],
        "amount": [1000, -500],
        "transaction_hour": [12, 14],
        "transaction_frequency": [2, 3],
        "location_changed": [0, 0],
        "device_changed": [0, 0],
    })

    cleaned = clean_transactions(data)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["amount"] == 1000
