# Customer Churn Prediction & Retention Analytics

An end-to-end supervised machine learning project that predicts telecom customer churn, benchmarks multiple classification algorithms, evaluates model reliability, and exposes predictions through a Streamlit application.

## Objective

Identify customers at higher risk of leaving so retention teams can prioritize interventions.

## Workflow

```text
Raw Customer Data
       ↓
Data Cleaning
       ↓
Train/Test Split
       ↓
Pipeline + ColumnTransformer
       ↓
Model Benchmarking
       ↓
Cross-Validation
       ↓
Metric Comparison
       ↓
Best Model
       ↓
Churn Probability
       ↓
Risk Segment
       ↓
Streamlit App
```

## Models

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine
- AdaBoost
- Gradient Boosting
- Soft Voting Ensemble

## Engineering

The preprocessing pipeline uses numeric median imputation and scaling, plus categorical mode imputation and one-hot encoding. Preprocessing is fitted inside the training pipeline to reduce leakage risk and keep inference consistent.

## Evaluation

The training workflow reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 5-fold stratified cross-validation ROC-AUC

For churn prediction, recall and ROC-AUC are evaluated alongside accuracy because missed churn cases can be costly.

## Dataset

The project is designed for the commonly used IBM Telco Customer Churn dataset. Download the CSV and place it at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Do not commit private or sensitive customer data.

## Project Structure

```text
Customer-Churn-Prediction/
├── app.py
├── data/
│   └── README.md
├── models/
├── outputs/
├── scripts/
│   └── train.py
├── src/
│   ├── churn_pipeline.py
│   └── visualize.py
├── .gitignore
├── LICENSE
├── QUICKSTART.md
├── PROJECT_SUMMARY.md
├── PROJECT_STRUCTURE.md
├── README.md
└── requirements.txt
```

## Run Locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add the dataset:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Train and evaluate:

```bash
python scripts/train.py
```

Launch the application:

```bash
streamlit run app.py
```

## Outputs

Training generates artifacts such as:

```text
outputs/model_comparison.csv
outputs/model_comparison.png
outputs/test_predictions.csv
models/best_churn_model.joblib
```

## Risk Segmentation

- High Risk: ≥ 70%
- Medium Risk: 40–69.9%
- Low Risk: < 40%

These are demonstration thresholds and should be calibrated using business costs before production use.

## Portfolio Highlights

This repository demonstrates:

- End-to-end supervised ML
- Leakage-aware preprocessing
- One-hot encoding and feature scaling
- Classical classification algorithms and ensembles
- Stratified cross-validation
- Model evaluation and comparison
- Probability-based churn risk scoring
- Model persistence with Joblib
- Interactive deployment with Streamlit

## Roadmap

- XGBoost benchmarking
- Randomized hyperparameter search
- Probability calibration
- SHAP explainability
- Cost-sensitive threshold optimization
- MLflow experiment tracking
- Docker deployment
- GitHub Actions CI
