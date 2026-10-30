@'
# Week 2 Model Report — Telemetry Anomaly Detection

## 1. Project Overview

This project detects anomalous patterns in telemetry data using machine learning. The objective is to build a reproducible, leakage-conscious evaluation pipeline, engineer domain-informed features, compare models, and document operational limitations.

## 2. Data and Evaluation

The selected Logistic Regression model was evaluated on 3,750 validation records, including 147 anomalies. The validation metrics below are reported separately from cross-validation results.

## 3. Models Evaluated

- Logistic Regression
- LightGBM
- XGBoost

The repository also contains baseline, ablation, and tuning results. See the corresponding CSV files in `reports/`.

## 4. Feature Engineering

The feature engineering experiment compared 9 original features with 17 features after engineering.

| Metric | Original features | Engineered features |
|---|---:|---:|
| ROC-AUC | 0.790966 | 0.801465 |
| PR-AUC | 0.211201 | 0.231269 |
| F1-score | 0.237474 | 0.262463 |
| Recall | 0.639416 | 0.648175 |

The engineered feature set improved ROC-AUC by 0.010499 and PR-AUC by 0.020068 in the reported cross-validation results.

The retry/error feature group produced a PR-AUC increase of 0.024368 over the original feature configuration. Log and time feature groups did not improve PR-AUC individually in these experiments.

## 5. Model Comparison and Tuning

Cross-validation results:

| Model | Mean ROC-AUC | Mean PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.801465 | 0.231269 |
| LightGBM | 0.757235 | 0.202652 |
| XGBoost | 0.757801 | 0.198325 |

Tuned boosting validation results:

| Model | Validation ROC-AUC | Validation PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.853952 | 0.275362 | 0.444444 | 0.027211 | 0.051282 |
| XGBoost | 0.845686 | 0.290474 | 0.440000 | 0.074830 | 0.127907 |

These results come from different evaluation summaries and should not be interpreted as a direct, statistically conclusive comparison. The saved Logistic Regression model uses a decision threshold of 0.85. Its recorded validation results are approximately 0.273 PR-AUC, 0.848 ROC-AUC, 0.31 precision, 0.35 recall, and 0.33 F1. The exact values should be checked against `final_model_validation.csv` and `final_threshold_comparison.csv` before deployment.

## 6. Error Analysis

For the saved model at the selected threshold:

- False positives: 114
- False negatives: 96

The repository includes `error_analysis_10_fp_10_fn.csv` and `validation_errors.csv` for detailed review. False positives can create unnecessary alerts; false negatives can allow anomalies to go undetected. The CSV findings should be reviewed to categorize root causes before making operational claims.

## 7. Explainability

The repository contains the following SHAP artifacts:

- `shap_feature_importance.png`
- `shap_summary.png`
- `shap_waterfall_1.png`
- `shap_waterfall_2.png`
- `shap_waterfall_3.png`
- `shap_feature_importance.csv`

These figures support global and individual prediction inspection. The plots should be examined before making claims about which features drive predictions.

## 8. Fairness and Subgroup Audit

A completed subgroup fairness audit is not established by the metrics currently summarized in this report. Before deployment, evaluate false-positive rate, false-negative rate, recall, and precision across available demographic or platform subgroups. If sensitive demographic attributes are unavailable, document that limitation and evaluate appropriate non-sensitive operational subgroups instead.

## 9. Operational Recommendations

- Treat the model as an experimental alerting component, not an autonomous decision-maker.
- Select the decision threshold according to the relative costs of missed anomalies and false alarms.
- Monitor precision, recall, PR-AUC, alert volume, and data drift after deployment.
- Review false positives and false negatives regularly and record root-cause categories.
- Revalidate the model when telemetry schemas, traffic patterns, or anomaly prevalence change.

## 10. Limitations

Anomaly recall and F1 depend on the selected model and decision threshold. Validation metrics are not final test-set metrics. The available results do not by themselves establish statistical significance, subgroup fairness, or production readiness.

## 11. Week 3 Transition Plan

1. Review and categorize the 10 false positives and 10 false negatives.
2. Complete a subgroup audit where suitable labels are available.
3. Validate the inference pipeline and threshold against a held-out evaluation set.
4. Add monitoring for prediction distributions, alert rates, and data drift.
5. Document deployment assumptions and define criteria for rollback or retraining.

## 12. Testing

The latest reported automated test run completed successfully: 28 tests passed with zero failures.
'@ | Set-Content -Path reports/week2_model_report.md -Encoding utf8
