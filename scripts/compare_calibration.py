import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import load_split_data

REPORTS_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "best_model.joblib"

_, _, X_val, y_val, _, _ = load_split_data()
artifact = joblib.load(MODEL_PATH)

base_model = artifact["model"]

calibrated_model = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=3,
)

calibrated_model.fit(*load_split_data()[:2])

models = {
    "original": base_model,
    "calibrated_sigmoid": calibrated_model,
}

results = []

for name, model in models.items():
    probabilities = model.predict_proba(X_val)[:, 1]

    results.append(
        {
            "model": name,
            "brier_score": brier_score_loss(y_val, probabilities),
            "log_loss": log_loss(y_val, probabilities),
            "roc_auc": roc_auc_score(y_val, probabilities),
            "pr_auc": average_precision_score(y_val, probabilities),
        }
    )

results_df = pd.DataFrame(results)
output_path = REPORTS_DIR / "calibration_model_comparison.csv"
results_df.to_csv(output_path, index=False)

print(results_df.to_string(index=False))
print(f"\nSaved comparison to: {output_path}")
