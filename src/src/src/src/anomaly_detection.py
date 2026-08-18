import pandas as pd
from sklearn.ensemble import IsolationForest


MODEL_FEATURES = [
    "log_amount",
    "transaction_frequency",
    "transaction_hour",
    "location_changed",
    "device_changed",
    "risk_signal",
]


def detect_anomalies(
    data: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, IsolationForest]:
    """
    Detect anomalous transactions using Isolation Forest.

    Parameters
    ----------
    data : pandas.DataFrame
        Feature-engineered transaction data.

    contamination : float
        Expected proportion of anomalous observations.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        DataFrame containing anomaly results and
        the trained Isolation Forest model.
    """

    missing = set(MODEL_FEATURES) - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    if not 0 < contamination < 0.5:
        raise ValueError(
            "contamination must be between 0 and 0.5"
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )

    X = data[MODEL_FEATURES].fillna(0)

    model.fit(X)

    result = data.copy()

    # Higher scores represent more normal observations.
    result["anomaly_score"] = model.decision_function(X)

    # Isolation Forest returns:
    #  1  = normal
    # -1  = anomaly
    result["anomaly_label"] = model.predict(X)

    result["is_anomaly"] = (
        result["anomaly_label"] == -1
    )

    return result, model
