from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_churn_model.joblib")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("Customer Churn Predictor")
st.caption("Portfolio ML application for telecom customer churn risk scoring")

if not MODEL_PATH.exists():
    st.warning("Trained model not found. Run `python scripts/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Customer information")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
    total = st.number_input("Total Charges", min_value=0.0, value=max(tenure * monthly, 0.0), step=10.0)

if st.button("Predict Churn", type="primary", use_container_width=True):
    row = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )

    probability = float(model.predict_proba(row)[0, 1])
    percentage = probability * 100

    if percentage >= 70:
        risk = "HIGH RISK"
        st.error(f"{risk} — churn probability: {percentage:.1f}%")
        st.write("Suggested action: prioritize a retention offer or proactive customer outreach.")
    elif percentage >= 40:
        risk = "MEDIUM RISK"
        st.warning(f"{risk} — churn probability: {percentage:.1f}%")
        st.write("Suggested action: monitor engagement and consider targeted retention messaging.")
    else:
        risk = "LOW RISK"
        st.success(f"{risk} — churn probability: {percentage:.1f}%")
        st.write("Suggested action: maintain service quality and normal engagement.")
