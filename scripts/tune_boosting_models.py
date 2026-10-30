from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_split_data, RANDOM_STATE


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, _, _ = load_split_data()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    models = {
        "LightGBM": (
            LGBMClassifier(
                random_state=RANDOM_STATE,
                verbosity=-1,
                n_jobs=1,
            ),
            {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [-1, 3, 5, 7, 10],
                "num_leaves": [15, 31, 50, 63],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "reg_lambda": [0, 0.1, 1, 5, 10],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=1,
            ),
            {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5, 7, 10],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1, 1, 5],
                "reg_lambda": [0.1, 1, 5, 10],
            },
        ),
    }

    summaries = []

    for model_name, (estimator, param_distributions) in models.items():
        print(f"\nTuning {model_name}...")

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=20,
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            error_score="raise",
            return_train_score=True,
        )

        search.fit(X_train, y_train)

        results = pd.DataFrame(search.cv_results_)
        results.to_csv(
            reports_dir / f"{model_name.lower()}_tuning_results.csv",
            index=False,
        )

        best_model = search.best_estimator_
        validation_probabilities = best_model.predict_proba(X_val)[:, 1]

        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            precision_score,
            recall_score,
            f1_score,
        )

        validation_predictions = (validation_probabilities >= 0.5).astype(int)

        metrics = {
            "model": model_name,
            "best_cv_pr_auc": search.best_score_,
            "validation_pr_auc": average_precision_score(
                y_val, validation_probabilities
            ),
            "validation_roc_auc": roc_auc_score(
                y_val, validation_probabilities
            ),
            "validation_precision": precision_score(
                y_val, validation_predictions, zero_division=0
            ),
            "validation_recall": recall_score(
                y_val, validation_predictions, zero_division=0
            ),
            "validation_f1": f1_score(
                y_val, validation_predictions, zero_division=0
            ),
            "best_params": str(search.best_params_),
        }

        summaries.append(metrics)

        print("Best parameters:", search.best_params_)
        print(f"Best CV PR-AUC: {search.best_score_:.4f}")
        print(f"Validation PR-AUC: {metrics['validation_pr_auc']:.4f}")
        print(f"Validation ROC-AUC: {metrics['validation_roc_auc']:.4f}")
        print(f"Validation F1: {metrics['validation_f1']:.4f}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        reports_dir / "boosting_tuning_comparison.csv",
        index=False,
    )

    print("\nSaved tuning results in reports/")
    print("reports/boosting_tuning_comparison.csv")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
