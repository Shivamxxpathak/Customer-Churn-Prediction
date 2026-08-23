# Project Summary

## Problem

Predict telecom customer churn and convert model probabilities into actionable retention-risk segments.

## Technical Approach

1. Clean and validate the dataset.
2. Split data using stratification.
3. Build leakage-aware preprocessing with `Pipeline` and `ColumnTransformer`.
4. Compare multiple supervised classification algorithms.
5. Evaluate with classification metrics and stratified cross-validation.
6. Select the strongest model by ROC-AUC.
7. Persist the trained pipeline with Joblib.
8. Serve predictions through Streamlit.

## Business Output

Each customer receives a churn probability and a demonstration risk segment:

- High: 70% and above
- Medium: 40% to 69.9%
- Low: below 40%

These thresholds are illustrative and should be tuned against retention economics for production use.
