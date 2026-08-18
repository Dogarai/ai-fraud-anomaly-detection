import pandas as pd


REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "amount",
    "transaction_hour",
    "transaction_frequency",
    "location_changed",
    "device_changed",
]


def validate_columns(data: pd.DataFrame) -> None:
    """Check that all required columns are present."""

    missing = set(REQUIRED_COLUMNS) - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def clean_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data.

    Removes duplicate transaction IDs and invalid transactions.
    """

    validate_columns(data)

    cleaned = data.copy()

    cleaned = cleaned.drop_duplicates(
        subset=["transaction_id"]
    )

    cleaned = cleaned[
        cleaned["amount"] >= 0
    ]

    cleaned = cleaned[
        cleaned["transaction_hour"].between(0, 23)
    ]

    cleaned = cleaned[
        cleaned["transaction_frequency"] > 0
    ]

    cleaned = cleaned.reset_index(drop=True)

    return cleaned
