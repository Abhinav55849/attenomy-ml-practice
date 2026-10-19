# Executive Insights Memo : Week 1 EDA Findings
**Author:** ML Engineering Intern  
**Program:** Attenomy ML Engineering Internship  
**Reviewed by:** Attenomy AI / ML Lead  
**Date:** Week 1 Final Submission  

---

## 1. Executive Summary
Provide a high-level summary (150-200 words) of the 25,000-row telemetry dataset audit, key risks identified, and core takeaways for Week 2 classical model development.

## 2. Dataset Hygiene & Anomaly Audit
- **Completeness:** Summarize missingness patterns across features.
- **Outlier Mitigation:** Detail how extreme session durations or login failure anomalies were handled.
- **Data Leakage Safeguards:** Confirm that preprocessing transformations fit solely on training data splits.

## 3. Core Statistical Findings & Hypothesis Testing
- **Hypothesis 1 (Chi-Square Test):** Statistical dependency between device OS / authentication channel and abuse rates.
- **Hypothesis 2 (Two-Sample Test):** Numeric distribution differences across normal and fraudulent user sessions.

## 4. Operational Risks & Mitigation for Week 2
- **Target Imbalance:** Target ratio observed in raw telemetry and planned sampling/weighting strategy.
- **Collinearity Risk:** High-correlation feature pairs identified and exclusion recommendations.
- **Model Readiness:** Confirmation that clean parquet dataset is ready for baseline logistic regression and tree ensembles.
