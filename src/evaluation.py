"""Cross-validation and baseline evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_model_pipeline,
)

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def evaluate_baselines_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare dummy baselines and classical models using stratified CV."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "dummy_stratified": DummyClassifier(
            strategy="stratified",
            random_state=RANDOM_STATE,
        ),
        "dummy_most_frequent": DummyClassifier(
            strategy="most_frequent",
        ),
        "logistic_regression": build_model_pipeline("logistic_regression"),
        "knn": build_model_pipeline("knn"),
        "decision_tree": build_model_pipeline("decision_tree"),
        "random_forest": build_model_pipeline("random_forest"),
    }

    results = []

    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )

        row = {"model": model_name}

        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)

        results.append(row)

    return pd.DataFrame(results).sort_values(
        "pr_auc_mean",
        ascending=False,
    ).reset_index(drop=True)


def evaluate_on_split(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float | list[list[int]]]:
    """Calculate classification metrics for a fitted model."""
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = model.decision_function(X)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
    }


def save_results(
    results: pd.DataFrame,
    output_path: str | Path = "reports/baseline_results.csv",
) -> Path:
    """Save evaluation results to a CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path


def _build_ablation_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Build a model pipeline for a specific feature configuration."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_feature_ablation_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare original features against engineered features using CV."""
    original_numeric = [
        "session_duration_sec",
        "payload_size_kb",
        "latency_ms",
        "retry_count",
    ]

    engineered_numeric = [
        "payload_per_second",
        "log_payload_size",
        "log_latency",
        "has_retries",
        "high_retry_count",
        "has_error",
        "hour_of_day",
        "is_weekend",
    ]

    categorical = list(CATEGORICAL_FEATURES)

    configurations = {
        "original_features": original_numeric,
        "engineered_features": original_numeric + engineered_numeric,
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for configuration, numeric_columns in configurations.items():
        columns = numeric_columns + categorical
        missing = sorted(set(columns) - set(X_train.columns))

        if missing:
            raise ValueError(
                f"Missing columns for {configuration}: {missing}"
            )

        model = _build_ablation_pipeline(
            numeric_features=numeric_columns,
            categorical_features=categorical,
        )

        scores = cross_validate(
            model,
            X_train[columns],
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            error_score="raise",
        )

        row = {
            "configuration": configuration,
            "feature_count": len(columns),
        }

        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))

        results.append(row)

    return pd.DataFrame(results)
