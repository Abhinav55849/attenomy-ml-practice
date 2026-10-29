"""Analyze incorrect predictions made by the tuned anomaly detection model."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_split_data, RANDOM_STATE, build_model_pipeline


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, _, _ = load_split_data()

    print("Training the tuned Logistic Regression model...")

    pipeline = build_model_pipeline("logistic_regression")
    classifier_name = pipeline.steps[-1][0]

    param_grid = {
        f"{classifier_name}__C": [0.01, 0.1, 1, 10, 100]
    }

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    print(f"\nBest parameters: {search.best_params_}")
    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    matrix_display = ConfusionMatrixDisplay.from_predictions(
        y_val,
        predictions,
        display_labels=["Normal", "Anomaly"],
        values_format="d",
    )
    matrix_display.ax_.set_title("Validation Confusion Matrix")
    matrix_display.figure_.tight_layout()
    matrix_display.figure_.savefig(
        reports_dir / "confusion_matrix.png",
        dpi=150,
    )
    plt.close(matrix_display.figure_)

    error_data = X_val.copy()
    error_data["actual"] = y_val.to_numpy()
    error_data["predicted"] = predictions
    error_data["anomaly_probability"] = probabilities

    errors = error_data[
        error_data["actual"] != error_data["predicted"]
    ].copy()

    errors["error_type"] = errors.apply(
        lambda row: (
            "False Positive"
            if row["predicted"] == 1
            else "False Negative"
        ),
        axis=1,
    )

    errors.to_csv(
        reports_dir / "validation_errors.csv",
        index=False,
    )

    error_counts = errors["error_type"].value_counts()
    error_counts.to_csv(
        reports_dir / "error_summary.csv",
        header=["count"],
    )

    print("\nPrediction errors:")
    print(error_counts.to_string())

    print("\nFiles saved:")
    print("reports/confusion_matrix.png")
    print("reports/validation_errors.csv")
    print("reports/error_summary.csv")


if __name__ == "__main__":
    main()
