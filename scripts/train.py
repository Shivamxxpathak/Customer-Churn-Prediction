from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.churn_pipeline import train_and_save  # noqa: E402
from src.visualize import save_model_comparison  # noqa: E402

DATASET = ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATASET_URLS = [
    "https://raw.githubusercontent.com/IBM/watsonx-ai-samples/master/cpd4.5/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv",
    "https://raw.githubusercontent.com/SaeidRostami/Customer_Churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def ensure_dataset() -> None:
    if DATASET.exists() and DATASET.stat().st_size > 100_000:
        return

    DATASET.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in DATASET_URLS:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            if b"customerID" not in response.content[:500]:
                raise ValueError("Downloaded file does not look like the expected Telco CSV.")
            DATASET.write_bytes(response.content)
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Unable to download the Telco churn dataset: {last_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment",
        action="store_true",
        help="Use the faster deployment training profile for Render.",
    )
    args = parser.parse_args()

    ensure_dataset()
    results, model_path = train_and_save(DATASET, deployment=args.deployment)
    save_model_comparison(results, ROOT_DIR / "outputs" / "model_comparison.png")

    print("\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model saved to: {model_path}")
