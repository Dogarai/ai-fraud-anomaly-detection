import pandas as pd


def summarize_anomalies(
    data: pd.DataFrame
) -> dict:
    """
    Calculate summary statistics for detected anomalies.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing an `is_anomaly` column.

    Returns
    -------
    dict
        Summary of anomaly detection results.
    """

    if "is_anomaly" not in data.columns:
        raise ValueError(
            "Data must contain an 'is_anomaly' column."
        )

    total = len(data)

    anomalies = int(
        data["is_anomaly"].sum()
    )

    anomaly_rate = (
        anomalies / total
        if total > 0
        else 0.0
    )

    return {
        "total_transactions": total,
        "anomalies_detected": anomalies,
        "anomaly_rate": anomaly_rate,
    }


def top_anomalies(
    data: pd.DataFrame,
    n: int = 20
) -> pd.DataFrame:
    """
    Return the most unusual transactions.

    Lower Isolation Forest anomaly scores
    represent more unusual observations.
    """

    if "anomaly_score" not in data.columns:
        raise ValueError(
            "Data must contain an 'anomaly_score' column."
        )

    if n <= 0:
        raise ValueError(
            "n must be greater than zero."
        )

    return (
        data.sort_values(
            "anomaly_score",
            ascending=True
        )
        .head(n)
    )
