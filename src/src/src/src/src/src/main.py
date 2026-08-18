from src.anomaly_detection import detect_anomalies
from src.data_cleaning import clean_transactions
from src.data_loader import (
    generate_transactions,
    save_transactions,
)
from src.evaluation import summarize_anomalies
from src.feature_engineering import create_features
from src.visualization import plot_transaction_amounts


def main():
    """Run the complete transaction anomaly detection pipeline."""

    print("Generating transaction data...")

    data = generate_transactions(
        n_transactions=5000
    )

    save_transactions(data)

    print("Cleaning transaction data...")

    data = clean_transactions(data)

    print("Creating machine-learning features...")

    data = create_features(data)

    print("Detecting anomalous transactions...")

    results, _ = detect_anomalies(data)

    summary = summarize_anomalies(results)

    print("\nDetection Summary")
    print("-----------------")

    for key, value in summary.items():
        print(f"{key}: {value}")

    results.to_csv(
        "results/anomaly_results.csv",
        index=False
    )

    plot_transaction_amounts(
        results
    )

    print("\nAnalysis completed successfully.")
    print(
        "Results saved to results/anomaly_results.csv"
    )


if __name__ == "__main__":
    main()
