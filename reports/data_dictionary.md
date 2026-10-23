\# Data Dictionary: Attenomy Telemetry (`clean.parquet`)



\*\*Source:\*\* `data/raw/attenomy\_telemetry\_raw.csv` — 25,000 rows, 13 columns.

\*\*Output:\*\* `data/clean/clean.parquet` — 25,000 rows, 13 columns, 0 missing values.



\## Data Cleaning and Leakage Prevention



The cleaned dataset contains 13 columns. No `split` column is included in the current exported Parquet file.



For machine learning, data splitting and preprocessing must be handled without data leakage. Any learned imputation values, outlier thresholds, encoding mappings, and scaling parameters should be fitted using the training data only and then applied to validation and test data.



| #  | Column                 | Dtype (clean)    | Business definition                                                                                     | Allowed range / values                                                                                                | Missing-value policy                                                                                                                                                                                   |

| -- | ---------------------- | ---------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

| 1  | `transaction\_id`       | string           | Unique identifier for one platform interaction.                                                         | `TRX\_` followed by 6 digits; expected to be unique.                                                                   | No missing values observed.                                                                                                                                                                            |

| 2  | `account\_id`           | string           | Identifies the customer account that made the call.                                                     | `ACT\_` followed by digits.                                                                                            | No missing values observed.                                                                                                                                                                            |

| 3  | `timestamp`            | datetime64 (UTC) | UTC timestamp indicating when the interaction occurred.                                                 | Valid UTC datetime.                                                                                                   | No missing values observed after cleaning.                                                                                                                                                             |

| 4  | `interaction\_type`     | string           | Functional category of the platform interaction.                                                        | `api\_query`, `doc\_extraction`, `agent\_orchestration`, `model\_inference`, `batch\_sync`.                                | No missing values observed. Categories are normalized.                                                                                                                                                 |

| 5  | `channel`              | string           | Indicates how the client connected to the platform.                                                     | `web\_app`, `python\_sdk`, `rest\_api`, `cli\_tool`.                                                                      | No missing values observed. Inconsistent casing is normalized.                                                                                                                                         |

| 6  | `session\_duration\_sec` | float64          | Duration of the interaction session in seconds.                                                         | Non-negative values; upper bound depends on the implemented outlier treatment.                                        | Originally, 972 values were missing and 60 values were negative. Negative values were treated as invalid. Missing values were imputed, and high values were capped according to the cleaning pipeline. |

| 7  | `payload\_size\_kb`      | float64          | Size of the request payload in kilobytes.                                                               | Observed raw range: approximately 0.31–917.07 KB.                                                                     | No missing values observed. Large values are retained because they may contain useful anomaly information.                                                                                             |

| 8  | `latency\_ms`           | float64          | Server-side response latency in milliseconds.                                                           | Observed raw range: approximately 10.0–4,706.1 ms.                                                                    | No missing values observed. High latency values are retained because they may indicate operational anomalies.                                                                                          |

| 9  | `retry\_count`          | int64            | Number of retry attempts for the interaction.                                                           | Integer values from 0 to 7 in the raw dataset.                                                                        | No missing values observed.                                                                                                                                                                            |

| 10 | `error\_code`           | string           | Error code associated with an unsuccessful interaction.                                                 | `NO\_ERROR`, `ERR\_AUTH\_EXPIRED`, `ERR\_RATE\_LIMIT`, `ERR\_PAYLOAD\_TOO\_LARGE`, `ERR\_UPSTREAM\_UNAVAILABLE`, `ERR\_TIMEOUT`. | Originally, 22,954 values were blank. According to the dataset documentation, blanks indicate successful execution and are normalized to `NO\_ERROR`.                                                   |

| 11 | `client\_tier`          | string           | Subscription tier associated with the customer.                                                         | `Starter`, `Pro`, `Enterprise`, `VIP`, `UNKNOWN`.                                                                     | Originally, 809 values were missing. Missing values are replaced with `UNKNOWN`.                                                                                                                       |

| 12 | `region`               | string           | Cloud deployment region that served the interaction.                                                    | `us-east-1`, `us-west-2`, `eu-central-1`, `ap-south-1`.                                                               | No missing values observed. Category formatting is normalized where required.                                                                                                                          |

| 13 | `is\_anomaly`           | int64            | Target label indicating whether the interaction was flagged as an operational risk or suspicious event. | `0` = normal; `1` = anomaly. Approximately 3.92% of raw records have label `1`.                                       | No missing values observed. This is the target variable for classification.                                                                                                                            |



\## Governance and Modeling Notes



1\. \*\*Potential label leakage:\*\* `error\_code` and `retry\_count` may be strongly associated with `is\_anomaly`. Their relationship with the target should be examined in the EDA notebook before model training.

2\. \*\*Train-only preprocessing:\*\* Any learned imputation values, outlier thresholds, encoders, and scaling parameters must be fitted on training data only and then applied to validation and test data.

3\. \*\*Categorical encoding:\*\* One-hot encoding can be considered for nominal variables. An ordinal encoding for `client\_tier` should only be used if its ordering is justified. The treatment of `UNKNOWN` must be defined explicitly.

4\. \*\*Feature selection:\*\* `transaction\_id` and `account\_id` should be evaluated carefully and generally excluded as direct predictive features unless a valid modeling justification exists.

5\. \*\*Source of definitions:\*\* Business definitions and raw schema information are based on `data/DATASET\_OVERVIEW.md`. Confirm implementation-specific cleaning details against `src/cleaning.py` and the actual exported Parquet file.

6\. \*\*Schema consistency:\*\* The exported dataset currently contains 13 columns and does not include a `split` column. Any future schema change should be reflected in this document.



