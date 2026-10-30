"""Analyze incorrect predictions made by the saved anomaly detection model."""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_split_data


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    artifact = joblib.load(model_path)

    model = artifact["model"]
    threshold = artifact["threshold"]

    _, _, X_val, y_val, _, _ = load_split_data()

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("Model:", artifact["model_name"])
    print("Threshold:", threshold)

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    matrix_display = ConfusionMatrixDisplay.from_predictions(
        y_val,
        predictions,
        labels=[0, 1],
        display_labels=["Normal", "Anomaly"],
        values_format="d",
    )
    matrix_display.ax_.set_title(
        f"Validation Confusion Matrix (threshold={threshold})"
    )
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
    error_data["threshold"] = threshold

    error_data["error_type"] = "Correct"
    error_data.loc[
        (error_data["actual"] == 0) & (error_data["predicted"] == 1),
        "error_type",
    ] = "False Positive"
    error_data.loc[
        (error_data["actual"] == 1) & (error_data["predicted"] == 0),
        "error_type",
    ] = "False Negative"

    errors = error_data[
        error_data["error_type"].isin(["False Positive", "False Negative"])
    ].copy()

    errors.to_csv(
        reports_dir / "validation_errors.csv",
        index=True,
        index_label="row_index",
    )

    selected_errors = []

    for error_type in ["False Positive", "False Negative"]:
        group = errors[errors["error_type"] == error_type].copy()

        if error_type == "False Positive":
            group = group.sort_values(
                "anomaly_probability",
                ascending=False,
            )
        else:
            group = group.sort_values(
                "anomaly_probability",
                ascending=False,
            )

        selected_errors.append(group.head(10))

    examples = pd.concat(selected_errors)

    examples.to_csv(
        reports_dir / "error_analysis_10_fp_10_fn.csv",
        index=True,
        index_label="row_index",
    )

    error_counts = errors["error_type"].value_counts()
    error_counts.rename_axis("error_type").to_frame(
        "count"
    ).to_csv(reports_dir / "error_summary.csv")

    print("\nError counts:")
    print(error_counts.to_string())

    print("\nSelected error examples:")
    print(
        examples[
            [
                "actual",
                "predicted",
                "anomaly_probability",
                "error_type",
            ]
        ].to_string()
    )

    print("\nSaved files:")
    print("reports/confusion_matrix.png")
    print("reports/validation_errors.csv")
    print("reports/error_summary.csv")
    print("reports/error_analysis_10_fp_10_fn.csv")


if __name__ == "__main__":
    main()
