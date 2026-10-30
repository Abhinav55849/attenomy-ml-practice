# Week 2 Model Report — Telemetry Anomaly Detection

## Project Overview

This project focuses on detecting anomalous patterns in telemetry data using machine learning.

## Models Evaluated

* Logistic Regression
* LightGBM
* XGBoost

## Selected Model

**Model:** Logistic Regression
**Decision Threshold:** 0.85

## Validation Metrics

The selected model was evaluated on 3,750 validation records.

| Metric            |            Result |
| ----------------- | ----------------: |
| Accuracy          | Approximately 94% |
| Anomaly Precision |              0.31 |
| Anomaly Recall    |              0.35 |
| Anomaly F1-score  |              0.33 |
| PR-AUC            |             0.273 |

## Error Analysis

* False Positives: 114
* False Negatives: 96

## Testing

Automated tests completed successfully: **28 passed, zero failures.**

## Limitations

The model has relatively low anomaly recall. These metrics are validation results, not final test-set results. Further evaluation is needed before production deployment.
