from pathlib import Path
import nbformat


def main():
    path = Path("notebooks/03_baselines.ipynb")

    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Refusing to overwrite it."
        )

    notebook = nbformat.v4.new_notebook()

    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Week 2 — Baseline Models\n\n"
            "Objective: Compare baseline models for anomaly detection.\n\n"
            "Use the existing dataset splits. Cross-validation uses "
            "training data only. Keep the test set untouched until "
            "model selection is complete."
        ),
        nbformat.v4.new_markdown_cell(
            "## 1. Load the Existing Dataset Splits"
        ),
        nbformat.v4.new_code_cell(
            "from src.pipeline import load_split_data\n\n"
            "X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()\n\n"
            'print("Train rows:", len(X_train))\n'
            'print("Validation rows:", len(X_val))\n'
            'print("Test rows:", len(X_test))\n'
            'print(f"Training anomaly rate: {y_train.mean():.2%}")'
        ),
        nbformat.v4.new_markdown_cell(
            "## 2. Target Distribution"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n\n"
            "target_summary = pd.DataFrame({\n"
            '    "split": ["train", "validation", "test"],\n'
            '    "rows": [len(y_train), len(y_val), len(y_test)],\n'
            '    "anomalies": [int(y_train.sum()), int(y_val.sum()), '
            'int(y_test.sum())],\n'
            "})\n"
            'target_summary["anomaly_rate"] = (\n'
            '    target_summary["anomalies"] / target_summary["rows"]\n'
            ")\n"
            "target_summary"
        ),
        nbformat.v4.new_markdown_cell(
            "## 3. Stratified 5-Fold Cross-Validation\n\n"
            "Preprocessing is kept inside each model pipeline. "
            "This prevents imputation, scaling, and encoding from "
            "learning from a validation fold during cross-validation."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "from src.evaluation import evaluate_baselines_cv\n\n"
            'results_path = Path("reports/baseline_results.csv")\n\n'
            "if results_path.exists():\n"
            "    results = pd.read_csv(results_path)\n"
            '    print("Loaded saved results:", results_path)\n'
            "else:\n"
            "    results = evaluate_baselines_cv(\n"
            "        X_train, y_train, n_splits=5\n"
            "    )\n"
            "    results_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    results.to_csv(results_path, index=False)\n"
            '    print("Saved results:", results_path)\n\n'
            'results.sort_values("pr_auc_mean", ascending=False).round(4)'
        ),
        nbformat.v4.new_markdown_cell(
            "## 4. Compare PR-AUC and ROC-AUC"
        ),
        nbformat.v4.new_code_cell(
            "import matplotlib.pyplot as plt\n\n"
            'plot_data = results.sort_values("pr_auc_mean")\n\n'
            'plot_data.plot.barh(\n'
            '    x="model", y="pr_auc_mean", legend=False, figsize=(9, 5)\n'
            ")\n"
            'plt.title("Baseline Comparison — PR-AUC")\n'
            'plt.xlabel("Mean PR-AUC")\n'
            "plt.tight_layout()\n"
            "plt.show()\n\n"
            'plot_data.plot.barh(\n'
            '    x="model", y="roc_auc_mean", legend=False, figsize=(9, 5)\n'
            ")\n"
            'plt.title("Baseline Comparison — ROC-AUC")\n'
            'plt.xlabel("Mean ROC-AUC")\n'
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "## 5. Cross-Validation Variability"
        ),
        nbformat.v4.new_code_cell(
            'results[["model", "roc_auc_std", "pr_auc_std"]].round(4)'
        ),
        nbformat.v4.new_markdown_cell(
            "## 6. Findings and Limitations\n\n"
            "- Logistic Regression currently has the highest mean PR-AUC "
            "among the tested models.\n"
            "- Accuracy alone is misleading for an imbalanced dataset.\n"
            "- KNN has very low recall at the default prediction threshold.\n"
            "- Compare CV standard deviations with the roadmap target of "
            "0.005; do not assume this criterion is satisfied.\n"
            "- These are baseline results, not final tuned-model results.\n"
            "- Do not use the test set to select models or thresholds."
        ),
    ]

    nbformat.validate(notebook)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)

    print(f"Created {path} with {len(notebook.cells)} cells.")


if __name__ == "__main__":
    main()
