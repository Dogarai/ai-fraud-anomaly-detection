from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_transaction_amounts(
    data: pd.DataFrame,
    output_path: str = "results/transaction_amounts.png",
) -> None:
    """
    Create and save a histogram of transaction amounts.

    Parameters
    ----------
    data : pandas.DataFrame
        Transaction data containing an `amount` column.

    output_path : str
        Location where the chart will be saved.
    """

    if "amount" not in data.columns:
        raise ValueError(
            "Data must contain an 'amount' column."
        )

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        data["amount"],
        bins=50
    )

    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.title("Transaction Amount Distribution")

    plt.tight_layout()

    plt.savefig(output)

    plt.close()
