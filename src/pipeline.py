"""Leakage-safe preprocessing and baseline model pipelines."""

from __future__ import annotations
from typing import Any
import pandas as pd
from src.features import add_features
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "session_duration_sec",
    "payload_size_kb",
    "latency_ms",
    "retry_count",
    "payload_per_second",
    "log_payload_size",
    "log_latency",
    "has_retries",
    "high_retry_count",
    "has_error",
    "hour_of_day",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "interaction_type",
    "channel",
    "error_code",
    "client_tier",
    "region",
]

TARGET_COLUMN = "is_anomaly"
SPLIT_COLUMN = "split"

EXCLUDED_COLUMNS = [
    "transaction_id",
    "account_id",
    "timestamp",
    TARGET_COLUMN,
    SPLIT_COLUMN,
]


def load_split_data(
    path: str = "data/clean/clean.parquet",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series]:
    """Load the existing train, validation, and test splits."""
    df = pd.read_parquet(path)
    df = add_features(df)

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET_COLUMN, SPLIT_COLUMN]
    )
    missing_columns = sorted(set(required_columns) - set(df.columns))

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"Target column '{TARGET_COLUMN}' contains missing values.")

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Target column '{TARGET_COLUMN}' must contain only 0 and 1.")

    expected_splits = {"train", "val", "test"}
    actual_splits = set(df[SPLIT_COLUMN].dropna().astype(str).unique())

    if actual_splits != expected_splits:
        raise ValueError(
            f"Expected splits {expected_splits}, but found {actual_splits}."
        )

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    split_data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    for split_name in ("train", "val", "test"):
        subset = df.loc[df[SPLIT_COLUMN].astype(str) == split_name]
        X = subset[feature_columns].copy()
        y = subset[TARGET_COLUMN].astype(int).copy()
        split_data[split_name] = (X, y)

    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["val"]
    X_test, y_test = split_data["test"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing steps that are fitted only on training data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_pipeline(model_name: str) -> Pipeline:
    """Build a complete preprocessing-plus-classifier pipeline."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            l1_ratio=0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    if model_name not in models:
        available = ", ".join(sorted(models))
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of: {available}"
        )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", models[model_name]),
        ]
    )


def build_dummy_pipeline(strategy: str = "stratified") -> Pipeline:
    """Build a simple baseline with the same leakage-safe preprocessing."""
    from sklearn.dummy import DummyClassifier

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DummyClassifier(strategy=strategy, random_state=RANDOM_STATE),
            ),
        ]
    )

