from pathlib import Path

from src.churn_pipeline import train_and_save
from src.visualize import save_model_comparison


DATASET = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


if __name__ == "__main__":
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET}. Add the IBM Telco Customer Churn CSV first."
        )

    results, model_path = train_and_save(DATASET)
    save_model_comparison(results, "outputs/model_comparison.png")

    print("\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model saved to: {model_path}")
    print("\nNext: streamlit run app.py")
