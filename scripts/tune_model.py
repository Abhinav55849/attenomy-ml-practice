"""Tune Logistic Regression using training data only."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_split_data, RANDOM_STATE
from src.evaluation import evaluate_on_split


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, _, _ = load_split_data()

    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("\nStarting Logistic Regression hyperparameter tuning...")

    from src.pipeline import build_model_pipeline

    pipeline = build_model_pipeline("logistic_regression")

    classifier_step = pipeline.steps[-1][0]

    param_grid = {
        f"{classifier_step}__C": [0.01, 0.1, 1, 10, 100],
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
        error_score="raise",
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(search.best_params_)

    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(
        reports_dir / "tuning_results.csv",
        index=False,
    )

    validation_metrics = evaluate_on_split(
        search.best_estimator_,
        X_val,
        y_val,
    )

    print("\nValidation metrics:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value}")

    pd.DataFrame([validation_metrics]).to_csv(
        reports_dir / "tuned_model_validation.csv",
        index=False,
    )

    print("\nSaved:")
    print("reports/tuning_results.csv")
    print("reports/tuned_model_validation.csv")


if __name__ == "__main__":
    main()
