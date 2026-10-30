"""Explain the anomaly detection model using SHAP."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import RANDOM_STATE, load_split_data


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = PROJECT_ROOT / "models" / "best_model.joblib"
    artifact = joblib.load(model_path)
    model = artifact["model"]
    threshold = artifact["threshold"]

    _, _, X_val, y_val, _, _ = load_split_data()

    print("Loaded saved model:", artifact["model_name"])
    print("Decision threshold:", threshold)

    sample_size = min(300, len(X_val))
    X_explain = X_val.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X_explain)
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=X_explain.index,
    )

    explainer = shap.LinearExplainer(
        classifier,
        X_transformed,
    )
    shap_values = explainer(X_transformed)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.abs(shap_values.values).mean(axis=0),
        }
    ).sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        reports_dir / "shap_feature_importance.csv",
        index=False,
    )

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        reports_dir / "shap_feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    shap.plots.beeswarm(
        shap_values,
        max_display=15,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        reports_dir / "shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    prediction_results = pd.DataFrame(
        {
            "row_index": X_val.index,
            "actual_label": y_val.to_numpy(),
            "predicted_probability": probabilities,
            "predicted_label": predictions,
        }
    )

    prediction_results["prediction_type"] = np.select(
        [
            (prediction_results["actual_label"] == 1)
            & (prediction_results["predicted_label"] == 1),
            (prediction_results["actual_label"] == 0)
            & (prediction_results["predicted_label"] == 0),
            (prediction_results["actual_label"] == 0)
            & (prediction_results["predicted_label"] == 1),
            (prediction_results["actual_label"] == 1)
            & (prediction_results["predicted_label"] == 0),
        ],
        [
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        ],
        default="unknown",
    )

    candidates = prediction_results[
        prediction_results["row_index"].isin(X_explain.index)
    ].copy()

    selected = []

    for category in [
        "true_positive",
        "false_positive",
        "false_negative",
    ]:
        matches = candidates[
            candidates["prediction_type"] == category
        ]

        if not matches.empty:
            selected.append(matches.iloc[0]["row_index"])

    for index in candidates.index:
        row_index = candidates.loc[index, "row_index"]

        if row_index not in selected:
            selected.append(row_index)

        if len(selected) == 3:
            break

    if len(selected) < 3:
        remaining = [
            idx for idx in X_explain.index
            if idx not in selected
        ]
        selected.extend(remaining[: 3 - len(selected)])

    selected = selected[:3]

    explanation_rows = []

    for plot_number, row_index in enumerate(selected, start=1):
        row_position = X_explain.index.get_loc(row_index)
        explanation = shap_values[row_position]

        shap.plots.waterfall(
            explanation,
            max_display=15,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            reports_dir / f"shap_waterfall_{plot_number}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        val_position = X_val.index.get_loc(row_index)

        explanation_rows.append(
            {
                "plot_number": plot_number,
                "row_index": row_index,
                "actual_label": int(y_val.iloc[val_position]),
                "predicted_probability": float(
                    probabilities[val_position]
                ),
                "threshold": float(threshold),
                "predicted_label": int(predictions[val_position]),
                "prediction_type": prediction_results.loc[
                    prediction_results["row_index"] == row_index,
                    "prediction_type",
                ].iloc[0],
            }
        )

    pd.DataFrame(explanation_rows).to_csv(
        reports_dir / "shap_waterfall_examples.csv",
        index=False,
    )

    print("\nTop 10 influential features:")
    print(importance.head(10).to_string(index=False))

    print("\nWaterfall examples:")
    print(pd.DataFrame(explanation_rows).to_string(index=False))

    print("\nSaved SHAP reports:")
    print("reports/shap_feature_importance.csv")
    print("reports/shap_feature_importance.png")
    print("reports/shap_summary.png")
    print("reports/shap_waterfall_1.png")
    print("reports/shap_waterfall_2.png")
    print("reports/shap_waterfall_3.png")
    print("reports/shap_waterfall_examples.csv")


if __name__ == "__main__":
    main()
