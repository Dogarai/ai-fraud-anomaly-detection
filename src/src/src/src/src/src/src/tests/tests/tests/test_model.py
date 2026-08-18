import pandas as pd
import pytest

from src.anomaly_detection import detect_anomalies


def create_test_data():
    return pd.DataFrame({
        "transaction_id": [
            f"TXN_{i:03d}"
            for i in range(20)
        ],
        "customer_id": [
            f"CUST_{i:03d}"
            for i in range(20)
        ],
        "amount": [10000] * 20,
        "transaction_hour": [12] * 20,
        "transaction_frequency": [2] * 20,
        "location_changed": [0] * 20,
        "device_changed": [0] * 20,
        "log_amount": [9.21] * 20,
        "risk_signal": [0] * 20,
    })


def test_model_creates_anomaly_columns():
    data = create_test_data()

    result, model = detect_anomalies(
        data,
        contamination=0.1
    )

    assert "anomaly_score" in result.columns
    assert "anomaly_label" in result.columns
    assert "is_anomaly" in result.columns


def test_model_returns_correct_number_of_rows():
    data = create_test_data()

    result, _ = detect_anomalies(
        data,
        contamination=0.1
    )

    assert len(result) == len(data)


def test_invalid_contamination_raises_error():
    data = create_test_data()

    with pytest.raises(ValueError):
        detect_anomalies(
            data,
            contamination=0.8
      )
