# Customer Churn Prediction & Retention Analytics

> **End-to-end ML application that predicts telecom customer churn risk and exposes the prediction workflow through a production-style Streamlit interface.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7)](https://render.com/)

## Live Application

**Render:** add the generated Render URL here after the first successful deployment.

The application accepts a customer profile, returns a churn probability, assigns a risk band, and provides a suggested retention action.

## Business Problem

Customer churn directly affects recurring revenue and customer lifetime value. The goal of this project is to estimate the probability that a telecom customer will churn so a retention team can prioritize proactive outreach.

This is a **decision-support prototype**, not an autonomous retention system. The 70% / 40% risk thresholds are demonstration thresholds and should be calibrated against real intervention costs and business capacity before production use.

## End-to-End Workflow

```text
IBM Telco Customer Churn Dataset
              ↓
Data Validation & Cleaning
              ↓
Numeric + Categorical Preprocessing
              ↓
Train / Test Split
              ↓
Multiple Classification Models
              ↓
ROC-AUC / F1 / Recall / Precision
              ↓
Best Model Selection
              ↓
Joblib Model Artifact
              ↓
Streamlit Prediction UI
              ↓
Churn Probability + Risk Band
              ↓
Retention Recommendation
```

## Models

The full local training profile benchmarks:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Machine
- AdaBoost
- Gradient Boosting

The Render deployment uses a **faster deployment profile** containing Logistic Regression, Random Forest, SVM and Gradient Boosting, then selects the best model by hold-out ROC-AUC.

## Engineering & ML Quality

- `ColumnTransformer` keeps preprocessing and model inference together.
- Numeric columns use median imputation and standardization.
- Categorical columns use most-frequent imputation and one-hot encoding.
- `handle_unknown="ignore"` makes inference robust to unseen categories.
- The target is stratified during train/test splitting.
- Evaluation includes Accuracy, Precision, Recall, F1 and ROC-AUC.
- The full local profile uses 5-fold stratified ROC-AUC cross-validation for the selected model.
- The trained pipeline is serialized with Joblib so the same preprocessing is reused at inference time.
- `customerID` is removed before training because it is an identifier, not a predictive feature.

## Dataset

The application uses the commonly used **IBM Telco Customer Churn** sample dataset containing customer demographics, services, contract information and billing information. The dataset is publicly available through IBM's sample-data repositories.

The training script automatically downloads the dataset when it is not already present locally. For local/manual runs, it can also be placed at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset is downloaded during Render's build process and is intentionally excluded from Git history.

## Repository Structure

```text
Customer-Churn-Prediction/
│
├── app.py                         # Streamlit application
├── render.yaml                    # Render Blueprint
├── requirements.txt               # Pinned dependencies
├── .streamlit/
│   └── config.toml                # Hosted Streamlit configuration
│
├── data/
│   └── README.md                  # Dataset instructions
│
├── models/                        # Generated model artifacts (gitignored)
│
├── outputs/                       # Generated metrics/plots (gitignored)
│
├── scripts/
│   └── train.py                   # Dataset bootstrap + training entry point
│
├── src/
│   ├── churn_pipeline.py          # Preprocessing, training, evaluation
│   └── visualize.py               # Model comparison visualization
│
├── PROJECT_STRUCTURE.md
├── PROJECT_SUMMARY.md
├── QUICKSTART.md
├── LICENSE
└── README.md
```

## Run Locally

### 1. Clone

```bash
git clone https://github.com/Shivamxxpathak/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train

The script downloads the dataset automatically if it is missing:

```bash
python scripts/train.py
```

For the faster deployment profile:

```bash
python scripts/train.py --deployment
```

### 5. Start the app

```bash
streamlit run app.py
```

## Render Deployment

This repository is prepared for Render as a Python web service.

### Render configuration

```text
Build Command:
pip install -r requirements.txt && python scripts/train.py --deployment

Start Command:
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT

Runtime:
Python 3.11.9

Region:
Singapore

Plan:
Free
```

The included `render.yaml` contains the deployment configuration for repeatable setup.

### Deployment behavior

1. Render clones the `main` branch.
2. Dependencies are installed.
3. The training script downloads the public dataset if needed.
4. Four deployment-profile models are trained.
5. The best model by ROC-AUC is saved to `models/best_churn_model.joblib`.
6. Model metadata and comparison results are generated.
7. Streamlit starts on Render's `$PORT`.
8. Future pushes to `main` trigger a new deployment.

## Generated Artifacts

```text
models/best_churn_model.joblib
models/model_metadata.json
outputs/model_comparison.csv
outputs/model_comparison.png
outputs/test_predictions.csv
```

These files are generated during training and are intentionally excluded from Git so the repository remains lightweight and reproducible.

## Application Features

### Prediction

- Customer demographic inputs
- Service and contract inputs
- Billing inputs
- Churn probability
- High / Medium / Low risk classification
- Retention recommendation

### Model Performance

The application exposes the generated model-comparison table and metric chart so users can see which model was selected.

## Risk Bands

| Churn probability | Risk | Example action |
|---:|---|---|
| ≥ 70% | High | Prioritize proactive retention outreach |
| 40–69.9% | Medium | Monitor engagement and consider targeted messaging |
| < 40% | Low | Maintain normal customer engagement |

These thresholds are **illustrative**, not a claim of production-calibrated risk.

## Portfolio Value

This project demonstrates practical skills across:

- Python
- Pandas / NumPy
- Scikit-learn
- Data preprocessing
- Feature engineering
- Classification
- Model comparison
- Cross-validation
- ROC-AUC / Precision / Recall / F1
- Model serialization
- Streamlit
- Render deployment
- Reproducible training

## Roadmap

Potential future upgrades:

- XGBoost benchmarking
- Hyperparameter optimization
- Probability calibration
- SHAP explainability
- Cost-sensitive threshold optimization
- MLflow experiment tracking
- Docker image deployment
- GitHub Actions CI/CD
- Monitoring and model drift checks

## License

MIT License.
