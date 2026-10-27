"""Run baseline cross-validation and save the comparison report."""

from src.evaluation import evaluate_baselines_cv, save_results
from src.pipeline import load_split_data


def main():
    print("Loading existing dataset splits...")
    X_train, y_train, _, _, _, _ = load_split_data()

    print(f"Training rows: {len(X_train)}")
    print(f"Training anomaly rate: {y_train.mean():.2%}")

    print("\nRunning stratified 5-fold cross-validation...")
    print("Evaluating 2 dummy baselines and 4 classical ML models.")
    print("This may take a few minutes.\n")

    results = evaluate_baselines_cv(X_train, y_train, n_splits=5)

    metrics = [
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
        "pr_auc_mean",
    ]

    print("Cross-validation results:")
    print(results[["model", *metrics]].to_string(index=False, float_format="%.4f"))

    output_path = save_results(results, "reports/baseline_results.csv")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
