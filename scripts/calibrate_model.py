import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from src.pipeline import load_split_data

REPORTS_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "best_model.joblib"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

_, _, X_val, y_val, _, _ = load_split_data()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]

probabilities = model.predict_proba(X_val)[:, 1]

brier = brier_score_loss(y_val, probabilities)
loss = log_loss(y_val, probabilities)

fraction_positive, mean_predicted = calibration_curve(
    y_val,
    probabilities,
    n_bins=10,
    strategy="quantile",
)

results = pd.DataFrame(
    {
        "mean_predicted_probability": mean_predicted,
        "fraction_of_positives": fraction_positive,
    }
)

results.to_csv(REPORTS_DIR / "calibration_results.csv", index=False)

plt.figure(figsize=(8, 6))
plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="Logistic Regression",
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of actual anomalies")
plt.title("Probability Calibration Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "calibration_curve.png", dpi=150)
plt.close()

summary = pd.DataFrame(
    [
        {
            "model": artifact["model_name"],
            "brier_score": brier,
            "log_loss": loss,
            "calibration_bins": len(results),
        }
    ]
)

summary.to_csv(REPORTS_DIR / "calibration_summary.csv", index=False)

print("Model:", artifact["model_name"])
print("Brier score:", round(brier, 6))
print("Log loss:", round(loss, 6))
print("\nCalibration results:")
print(results.to_string(index=False))
print("\nSaved reports:")
print(REPORTS_DIR / "calibration_results.csv")
print(REPORTS_DIR / "calibration_summary.csv")
print(REPORTS_DIR / "calibration_curve.png")
