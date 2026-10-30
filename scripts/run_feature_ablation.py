from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import load_split_data
from src.evaluation import evaluate_feature_ablation_cv


def main():
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _, _, _, _ = load_split_data()

    results = evaluate_feature_ablation_cv(
        X_train,
        y_train,
        n_splits=5,
    )

    output_path = reports_dir / "feature_group_ablation.csv"
    results.to_csv(output_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()