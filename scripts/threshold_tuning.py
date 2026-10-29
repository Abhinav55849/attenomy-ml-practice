"""Find a suitable prediction threshold for anomaly detection."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_split_data, RANDOM_STATE, build_model_pipeline


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, _, _ = load_split_data()

    print("Training Logistic Regression and selecting its C value...")

    pipeline = build_model_pipeline("logistic_regression")
    classifier_name = pipeline.steps[-1][0]

    param_grid = {
        f"{classifier_name}__C": [0.01, 0.1, 1, 10, 100]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    print(f"Best parameters: {search.best_params_}")

    probabilities = model.predict_proba(X_val)[:, 1]

    thresholds = [round(value, 2) for value in
                  [0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                   0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                   0.70, 0.75, 0.80, 0.85, 0.90]]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),
            "predicted_anomalies": int(predictions.sum()),
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        reports_dir / "threshold_results.csv",
        index=False,
    )

    best_row = results_df.loc[results_df["f1"].idxmax()]

    print("\nThreshold comparison:")
    print(results_df.round(4).to_string(index=False))

    print("\nBest threshold by validation F1-score:")
    print(f"Threshold: {best_row['threshold']:.2f}")
    print(f"Precision: {best_row['precision']:.4f}")
    print(f"Recall: {best_row['recall']:.4f}")
    print(f"F1-score: {best_row['f1']:.4f}")
    print(f"Predicted anomalies: {int(best_row['predicted_anomalies'])}")

    plt.figure(figsize=(9, 5))
    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        marker="o",
        label="Precision",
    )
    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        marker="o",
        label="Recall",
    )
    plt.plot(
        results_df["threshold"],
        results_df["f1"],
        marker="o",
        label="F1-score",
    )
    plt.xlabel("Prediction threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning on Validation Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        reports_dir / "threshold_comparison.png",
        dpi=150,
    )
    plt.close()

    print("\nSaved:")
    print("reports/threshold_results.csv")
    print("reports/threshold_comparison.png")


if __name__ == "__main__":
    main()
