"""Feature engineering utilities for anomaly detection."""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features without modifying the input DataFrame."""
    features = df.copy()

    duration = features["session_duration_sec"].clip(lower=1)
    features["payload_per_second"] = (
        features["payload_size_kb"] / duration
    )

    features["log_payload_size"] = np.log1p(
        features["payload_size_kb"].clip(lower=0)
    )
    features["log_latency"] = np.log1p(
        features["latency_ms"].clip(lower=0)
    )

    features["has_retries"] = (
        features["retry_count"].fillna(0) > 0
    ).astype("int8")

    features["high_retry_count"] = (
        features["retry_count"].fillna(0) >= 3
    ).astype("int8")

    error_code = features["error_code"]
    features["has_error"] = (
        error_code.notna()
        & error_code.astype(str).str.strip().ne("")
        & error_code.astype(str).str.lower().ne("none")
    ).astype("int8")

    # 7. Hour of day from the timestamp.
    timestamp = pd.to_datetime(
        features["timestamp"], errors="coerce", utc=True
    )
    features["hour_of_day"] = timestamp.dt.hour

    features["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype("int8")

    return features
