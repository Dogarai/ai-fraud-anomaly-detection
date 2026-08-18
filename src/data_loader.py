from pathlib import Path

import numpy as np
import pandas as pd


def generate_transactions(
    n_transactions: int = 5000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic financial transaction dataset.

    Parameters
    ----------
    n_transactions : int
        Number of transactions to generate.
    random_state : int
        Seed used for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Synthetic transaction data.
    """

    rng = np.random.default_rng(random_state)

    customer_ids = [
        f"CUST_{i:04d}"
        for i in rng.integers(1, 1001, n_transactions)
    ]

    amounts = rng.lognormal(
        mean=np.log(15000),
        sigma=0.8,
        size=n_transactions
    )

    hours = rng.integers(0, 24, n_transactions)

    frequencies = rng.poisson(
        lam=3,
        size=n_transactions
    ) + 1

    locations_changed = rng.binomial(
        n=1,
        p=0.08,
        size=n_transactions
    )

    devices_changed = rng.binomial(
        n=1,
        p=0.10,
        size=n_transactions
    )

    data = pd.DataFrame(
        {
            "transaction_id": [
                f"TXN_{i:06d}"
                for i in range(1, n_transactions + 1)
            ],
            "customer_id": customer_ids,
            "amount": amounts.round(2),
            "transaction_hour": hours,
            "transaction_frequency": frequencies,
            "location_changed": locations_changed,
            "device_changed": devices_changed,
        }
    )

    return data


def save_transactions(
    data: pd.DataFrame,
    output_path: str = "data/transactions.csv"
) -> None:
    """Save transaction data to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(path, index=False)
