from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_model_comparison(results: pd.DataFrame, output_path: str | Path) -> None:
    """Save a clean horizontal bar chart comparing ROC-AUC by model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = results.sort_values("ROC-AUC", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(ordered["Model"], ordered["ROC-AUC"])
    plt.xlabel("ROC-AUC")
    plt.title("Customer Churn Model Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
