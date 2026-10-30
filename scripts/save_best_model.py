from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    RANDOM_STATE,
    build_model_pipeline,
    load_split_data,
)


def main():
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, _, _ = load_split_data()

    pipeline = build_model_pipeline("logistic_regression")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        pipeline,
        {"classifier__C": [0.01, 0.1, 1, 10, 100]},
        scoring="average_precision",
        cv=cv,
        n_jobs=1,
        refit=True,
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    val_probabilities = model.predict_proba(X_val)[:, 1]

    thresholds = [0.50, 0.85]
    threshold_results = []

    for threshold in thresholds:
        predictions = (val_probabilities >= threshold).astype(int)

        threshold_results.append({
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
        })

    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(
        reports_dir / "final_threshold_comparison.csv",
        index=False,
    )

    selected_threshold = float(
        threshold_df.loc[threshold_df["f1"].idxmax(), "threshold"]
    )

    artifact = {
        "model": model,
        "threshold": selected_threshold,
        "model_name": "Logistic Regression",
        "best_params": search.best_params_,
        "cv_pr_auc": search.best_score_,
    }

    joblib.dump(artifact, models_dir / "best_model.joblib")

    val_predictions = (
        val_probabilities >= selected_threshold
    ).astype(int)

    metrics = {
        "model": "Logistic Regression",
        "best_params": str(search.best_params_),
        "threshold": selected_threshold,
        "cv_pr_auc": search.best_score_,
        "validation_precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "validation_recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "validation_f1": f1_score(
            y_val, val_predictions, zero_division=0
        ),
        "validation_pr_auc": average_precision_score(
            y_val, val_probabilities
        ),
        "validation_roc_auc": roc_auc_score(
            y_val, val_probabilities
        ),
    }

    pd.DataFrame([metrics]).to_csv(
        reports_dir / "final_model_validation.csv",
        index=False,
    )

    print("Best parameters:", search.best_params_)
    print("Selected threshold:", selected_threshold)
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved model: models/best_model.joblib")
    print("Saved metrics: reports/final_model_validation.csv")
    print("Saved thresholds: reports/final_threshold_comparison.csv")


if __name__ == "__main__":
    main()
