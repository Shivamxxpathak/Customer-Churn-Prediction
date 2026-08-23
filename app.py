from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    .risk-box {padding: 1.2rem; border-radius: 14px; margin: 1rem 0;}
    .metric-card {padding: 1rem; border: 1px solid rgba(128,128,128,.2); border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


st.title("📊 Customer Churn Prediction")
st.caption("End-to-end machine learning application for telecom churn risk scoring")

if not MODEL_PATH.exists():
    st.error("The trained model is not available. Run `python scripts/train.py` before starting the app.")
    st.stop()

model = load_model()
metadata = load_metadata()

with st.sidebar:
    st.header("About the model")
    st.write("The application uses the best-performing model selected during training by ROC-AUC.")
    if metadata:
        st.metric("Selected model", metadata.get("best_model", "—"))
        metrics = metadata.get("metrics", {})
        if "ROC-AUC" in metrics:
            st.metric("Test ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
        if "F1" in metrics:
            st.metric("Test F1", f"{metrics['F1']:.3f}")
    st.divider()
    st.caption("Built with Python, pandas, scikit-learn and Streamlit.")

predict_tab, model_tab, about_tab = st.tabs(["Predict Churn", "Model Performance", "Project"])

with predict_tab:
    st.subheader("Customer profile")
    st.info("Enter customer details and estimate the probability that the customer will churn.")

    left, right = st.columns(2)

    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with right:
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
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(max(tenure * monthly, 0.0)),
            step=10.0,
        )

    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
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

        st.divider()
        st.subheader("Prediction result")
        metric_col, result_col = st.columns([1, 2])
        with metric_col:
            st.metric("Churn probability", f"{percentage:.1f}%")
            st.progress(probability)
        with result_col:
            if percentage >= 70:
                st.error("🔴 HIGH RISK")
                st.write("Prioritize proactive retention outreach and review the customer's plan, pricing, and service experience.")
            elif percentage >= 40:
                st.warning("🟠 MEDIUM RISK")
                st.write("Monitor engagement and consider targeted retention messaging or a service review.")
            else:
                st.success("🟢 LOW RISK")
                st.write("Maintain normal engagement and service quality.")

with model_tab:
    st.subheader("Model comparison")
    comparison_path = BASE_DIR / "outputs" / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        chart_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"] if c in comparison.columns]
        if chart_cols:
            st.bar_chart(comparison.set_index("Model")[chart_cols])
    else:
        st.info("Model comparison results will appear after training.")

with about_tab:
    st.subheader("Project overview")
    st.markdown(
        """
        **Customer Churn Prediction & Retention Analytics** is an end-to-end supervised machine-learning project.

        **Pipeline**
        1. Data cleaning and validation
        2. Numeric imputation and standardization
        3. Categorical imputation and one-hot encoding
        4. Multiple classification models
        5. Hold-out evaluation using Accuracy, Precision, Recall, F1 and ROC-AUC
        6. Best-model selection and serialization
        7. Interactive customer-level prediction

        **Risk bands**
        - **High:** 70% or higher
        - **Medium:** 40%–69.9%
        - **Low:** below 40%
        """
    )
