"""Explain the anomaly detection model using SHAP."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    RANDOM_STATE,
    build_model_pipeline,
    load_split_data,
)


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, _, _ = load_split_data()

    print("Training Logistic Regression model...")

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
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    print(f"Best parameters: {search.best_params_}")
    print("Calculating SHAP values...")

    sample_size = min(300, len(X_val))
    X_explain = X_val.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps[classifier_name]

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

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": abs(shap_values.values).mean(axis=0),
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False,
    )

    importance.to_csv(
        reports_dir / "shap_feature_importance.csv",
        index=False,
    )

    print("\nTop 10 influential features:")
    print(importance.head(10).to_string(index=False))

    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        reports_dir / "shap_feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        reports_dir / "shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("\nSaved SHAP reports:")
    print("reports/shap_feature_importance.csv")
    print("reports/shap_feature_importance.png")
    print("reports/shap_summary.png")


if __name__ == "__main__":
    main()
