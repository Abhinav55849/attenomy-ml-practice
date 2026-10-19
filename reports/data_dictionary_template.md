# Data Governance Dictionary : Attenomy Telemetry Benchmark
**Schema Version:** 1.0.0  
**Target Variable:** `is_fraud_or_abuse` (Binary: 0 = Benign, 1 = Fraudulent/Abusive)

| Column Name | Data Type | Null Policy | Description | Allowable Range / Categories |
| :--- | :--- | :--- | :--- | :--- |
| `event_id` | String | Disallow | Unique telemetry event identifier | UUIDv4 |
| `timestamp` | Datetime | Disallow | ISO-8601 UTC event generation timestamp | 2026-01-01 to 2026-10-15 |
| `user_id` | String | Disallow | Anonymized user account identifier | USR_XXXXX |
| `device_os` | String | 'UNKNOWN' | Operating system of user device | Android, iOS, Windows, macOS, Linux, UNKNOWN |
| `ip_address` | String | 'UNKNOWN' | Anonymized client IP address | IPv4 representation |
| `location_country` | String | 'UNKNOWN' | 2-letter ISO country code | US, IN, GB, DE, SG, CA, UNKNOWN |
| `session_duration_sec`| Float | Median | Duration of user session in seconds | >= 0 |
| `login_failures_last_24h`| Integer | 0 | Count of unsuccessful logins in prior 24h | >= 0 |
| `api_call_count` | Integer | Median | Total API calls dispatched during session | >= 1 |
| `request_payload_bytes`| Integer | Median | Total payload size transmitted | >= 0 |
| `is_vpn_or_proxy` | Integer | 0 | Binary flag indicating VPN or anonymizing proxy | 0, 1 |
| `risk_score_external`| Float | Median | Third-party vendor risk scoring | 0.00 to 1.00 |
| `is_fraud_or_abuse` | Integer | Disallow | Target ground-truth abuse label | 0, 1 |
