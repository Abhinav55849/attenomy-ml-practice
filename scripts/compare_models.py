from pathlib import Path
import sys

import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate

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

    X_train, y_train, _, _, _, _ = load_split_data()

    models = {
        "Logistic Regression": build_model_pipeline("logistic_regression"),
        "LightGBM": build_model_pipeline("logistic_regression"),
        "XGBoost": build_model_pipeline("logistic_regression"),
    }

    models["LightGBM"].steps[-1] = (
        "classifier",
        LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=1,
        ),
    )

    models["XGBoost"].steps[-1] = (
        "classifier",
        XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining and evaluating: {name}")

        scores = cross_validate(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        results.append(row)

        print(f"PR-AUC: {row['pr_auc_mean']:.4f}")
        print(f"F1: {row['f1_mean']:.4f}")
        print(f"Recall: {row['recall_mean']:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        "pr_auc_mean",
        ascending=False,
    )

    output_path = reports_dir / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFinal model comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {output_path}")

if __name__ == "__main__":
    main()
