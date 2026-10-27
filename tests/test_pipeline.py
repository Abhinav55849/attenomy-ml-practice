"""Tests for the leakage-safe machine learning pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.features import add_features
from src.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_dummy_pipeline,
    build_model_pipeline,
    build_preprocessor,
    load_split_data,
)

@pytest.fixture
def sample_features():
    """Create sample data with engineered features."""
    raw_features = pd.DataFrame(
        {
            "session_duration_sec": [
                10.0, 20.0, 30.0, 40.0,
                50.0, 60.0, 70.0, 80.0,
            ],
            "payload_size_kb": [
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
            ],
            "latency_ms": [
                100.0, 200.0, 150.0, 300.0,
                120.0, 250.0, 180.0, 350.0,
            ],
            "retry_count": [0, 1, 0, 2, 1, 3, 0, 2],
            "interaction_type": ["read", "write"] * 4,
            "channel": ["web", "mobile"] * 4,
            "error_code": ["UNKNOWN", "E101", "UNKNOWN", "E202"] * 2,
            "client_tier": ["basic", "premium"] * 4,
            "region": ["north", "south"] * 4,
            "timestamp": [
                "2026-10-19T10:00:00Z",
                "2026-10-20T11:00:00Z",
                "2026-10-21T12:00:00Z",
                "2026-10-22T13:00:00Z",
                "2026-10-23T14:00:00Z",
                "2026-10-24T15:00:00Z",
                "2026-10-25T16:00:00Z",
                "2026-10-26T17:00:00Z",
            ],
        }
    )

    engineered_features = add_features(raw_features)

    return engineered_features.drop(columns=["timestamp"])

@pytest.fixture
def sample_target():
    """Create a binary target with both classes."""
    return pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="is_anomaly")


def test_preprocessor_uses_expected_features():
    """Ensure all expected numeric and categorical features are configured."""
    preprocessor = build_preprocessor()

    configured_columns = [
        column
        for _, _, columns in preprocessor.transformers
        for column in columns
    ]

    assert set(configured_columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_preprocessor_transforms_sample_data(sample_features):
    """Ensure preprocessing produces a numeric feature matrix."""
    transformed = build_preprocessor().fit_transform(sample_features)

    assert transformed.shape[0] == len(sample_features)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert np.isfinite(transformed).all()


@pytest.mark.parametrize(
    "model_name",
    ["logistic_regression", "knn", "decision_tree", "random_forest"],
)
def test_model_pipeline_fits_and_predicts(model_name, sample_features, sample_target):
    """Ensure each classical model can fit and produce binary predictions."""
    model = build_model_pipeline(model_name)
    model.fit(sample_features, sample_target)

    predictions = model.predict(sample_features)

    assert len(predictions) == len(sample_target)
    assert set(predictions).issubset({0, 1})


def test_model_pipeline_ignores_unseen_categories(sample_features, sample_target):
    """Ensure unseen categorical values do not break inference."""
    model = build_model_pipeline("logistic_regression")
    model.fit(sample_features, sample_target)

    new_sample = sample_features.iloc[[0]].copy()
    new_sample.loc[:, "region"] = "previously_unseen_region"

    prediction = model.predict(new_sample)

    assert len(prediction) == 1


def test_unknown_model_name_raises_error():
    """Ensure unsupported model names produce a clear error."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_model_pipeline("not_a_real_model")


def test_dummy_pipeline_fits_and_predicts(sample_features, sample_target):
    """Ensure the stratified dummy baseline can fit and predict."""
    model = build_dummy_pipeline(strategy="stratified")
    model.fit(sample_features, sample_target)

    predictions = model.predict(sample_features)

    assert len(predictions) == len(sample_target)


def test_load_split_data_preserves_existing_split_sizes():
    """Ensure the existing train, validation, and test split sizes are preserved."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()

    assert len(X_train) == len(y_train) == 17_500
    assert len(X_val) == len(y_val) == 3_750
    assert len(X_test) == len(y_test) == 3_750

    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_val.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})

    assert list(X_train.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_model_pipeline_is_a_sklearn_pipeline():
    """Ensure preprocessing and classification are combined in one pipeline."""
    model = build_model_pipeline("random_forest")

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

