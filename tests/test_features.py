"""Tests for feature engineering utilities."""
import pandas as pd
from src.features import add_features

def sample_data():
    """Create sample transaction data for feature engineering tests."""
    return pd.DataFrame(
{
"session_duration_sec": [10, 0, 20, 5],
"payload_size_kb": [100, 50, 200, 0],
"latency_ms": [20, 100, 50, 0],
"retry_count": [0, 1, 3, None],
"error_code": [None, "E101", "UNKNOWN", ""],
"timestamp": [
"2026-10-26 10:00:00",
"2026-10-24 12:00:00",
"2026-10-25 15:00:00",
"invalid-timestamp",
],
}
)

def test_add_features_creates_expected_columns():
    """Ensure all eight engineered features are created."""
    result = add_features(sample_data())


    expected = {
        "payload_per_second",
        "log_payload_size",
        "log_latency",
        "has_retries",
        "high_retry_count",
        "has_error",
        "hour_of_day",
        "is_weekend",
    }

    assert expected.issubset(result.columns)


def test_add_features_preserves_rows_and_original_data():
    """Ensure feature engineering does not mutate the input or remove rows."""
    df = sample_data()
    original = df.copy(deep=True)

    result = add_features(df)

    assert len(result) == len(df)
    pd.testing.assert_frame_equal(df, original)


def test_payload_per_second_handles_zero_duration():
    """Ensure zero duration does not cause division by zero."""
    result = add_features(sample_data())
    assert result.loc[1, "payload_per_second"] == 50

def test_retry_features():
    """Ensure retry indicator features follow their defined rules."""
    result = add_features(sample_data())


    assert result["has_retries"].tolist() == [0, 1, 1, 0]
    assert result["high_retry_count"].tolist() == [0, 0, 1, 0]

def test_timestamp_features_handle_invalid_values():
    """Ensure valid timestamps produce time features and invalid ones are missing."""
    result = add_features(sample_data())


    assert result.loc[0, "hour_of_day"] == 10
    assert result.loc[1, "is_weekend"] == 1
    assert result.loc[2, "is_weekend"] == 1
    assert pd.isna(result.loc[3, "hour_of_day"])
    assert result.loc[3, "is_weekend"] == 0
