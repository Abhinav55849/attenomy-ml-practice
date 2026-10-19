# Attenomy Telemetry Benchmark Dataset (Raw)

This dataset contains raw, unprocessed platform interaction records from enterprise systems. It serves as the primary benchmark dataset for **Week 1 (Data Cleaning & EDA)** and **Week 2 (Classical Machine Learning & Pipeline Engineering)**.

---

## Dataset Summary

- **Total Rows**: 25,000
- **Total Columns**: 13
- **Anomaly Target Prevalence (`is_anomaly`)**: 3.92% (positive class)
- **Primary File**: [`attenomy_telemetry_raw.csv`](./attenomy_telemetry_raw.csv)

---

## Schema & Column Definitions

| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `transaction_id` | String | 0 | Unique alphanumeric transaction identifier (`TRX_XXXXXX`). |
| `account_id` | String | 0 | Customer account identifier (`ACT_XXXX`). |
| `timestamp` | ISO-8601 | 0 | Request UTC timestamp string (`YYYY-MM-DDTHH:MM:SSZ`). |
| `interaction_type` | Categorical | 0 | Functional category (`api_query`, `doc_extraction`, `agent_orchestration`, `model_inference`, `batch_sync`). |
| `channel` | Categorical | 0 | Request origin client. Contains inconsistent casing (`web_app`, `WEB_APP`, `python_sdk`, `Python_SDK`, `rest_api`, `cli_tool`) to clean in Task 1.2. |
| `session_duration_sec` | Float | 972 | Duration of the session in seconds. Contains missing values and corrupted negative entries requiring imputation and filtering. |
| `payload_size_kb` | Float | 0 | Request payload size in kilobytes. |
| `latency_ms` | Float | 0 | Server-side round-trip response latency in milliseconds. Exhibits long-tail skew and operational latency spikes. |
| `retry_count` | Integer | 0 | Number of retry attempts executed for this interaction (range: 0 to 7). |
| `error_code` | Categorical | 22,954 | Specific error string when an interaction failed (`ERR_TIMEOUT`, `ERR_RATE_LIMIT`, etc.). Blank on successful execution. |
| `client_tier` | Categorical | 809 | Customer subscription tier (`Starter`, `Pro`, `Enterprise`, `VIP`). Contains ~3% blank entries requiring imputation. |
| `region` | Categorical | 0 | Cloud deployment region (`us-east-1`, `us-west-2`, `eu-central-1`, `ap-south-1`). |
| `is_anomaly` | Binary (0/1) | 0 | **Target Label**: 1 if flagged as an operational risk or suspicious interaction, 0 otherwise. |

---

## Tasks for Interns

1. **Week 1 Task 1.2 & 1.3**:
   - Ingest `attenomy_telemetry_raw.csv`.
   - Normalize string casing across `channel`.
   - Detect and handle corrupted negative values in `session_duration_sec`.
   - Defensively impute missing values (fit imputers strictly on training folds).
   - Generate 8 EDA figures and 2 statistical hypothesis tests.
   - Export cleaned dataset to `data/clean/clean.parquet`.

2. **Week 2 Modeling**:
   - Train baseline and tuned classification models targeting `is_anomaly`.
   - Evaluate with Stratified 5-Fold Cross-Validation (PR-AUC, ROC-AUC, F1-Score).
   - Compute TreeSHAP feature importances to interpret model decisions.
