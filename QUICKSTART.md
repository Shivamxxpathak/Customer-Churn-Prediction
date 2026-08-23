# Quick Start

## 1. Clone

```bash
git clone https://github.com/Shivamxxpathak/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

## 2. Create environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

## 3. Install

```bash
pip install -r requirements.txt
```

## 4. Add data

Place the IBM Telco Customer Churn CSV at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## 5. Train

```bash
python scripts/train.py
```

## 6. Run app

```bash
streamlit run app.py
```

The training command writes the model to `models/best_churn_model.joblib` and evaluation outputs to `outputs/`.
