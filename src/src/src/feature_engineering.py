import numpy as np
import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create machine-learning features from transaction data.

    Parameters
    ----------
    data : pandas.DataFrame
        Clean transaction data.

    Returns
    -------
    pandas.DataFrame
        Transaction data with additional features.
    """

    features = data.copy()

    # Reduce the effect of extremely large transaction amounts.
    features["log_amount"] = np.log1p(
        features["amount"]
    )

    # Transactions between 11 PM and 6 AM are treated
    # as nighttime transactions.
    features["is_night"] = (
        (features["transaction_hour"] < 6)
        | (features["transaction_hour"] >= 23)
    ).astype(int)

    # Combine several behavioral signals.
    features["risk_signal"] = (
        features["location_changed"]
        + features["device_changed"]
        + features["is_night"]
    )

    return features
