import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "credit_risk_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="centered"
)

st.title("🏦 Credit Risk Prediction System")
st.write("Predict whether a loan applicant is **High Risk** or **Low Risk**")

st.divider()

# ----------------------------------
# User input form
# ----------------------------------
with st.form("loan_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0)
        loan_term = st.number_input("Loan Term (months)", min_value=0.0)
        credit_history = st.selectbox("Credit History", [1, 0])

    submit = st.form_submit_button("Predict Risk")

# ----------------------------------
# Prediction logic
# ----------------------------------
if submit:

    total_income = applicant_income + coapplicant_income

    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Total_Income": total_income
    }

    df = pd.DataFrame([input_data])

    # Feature engineering
    df["Loan_Income_Ratio"] = (
        df["LoanAmount"] / df["Total_Income"]
        if df["Total_Income"].iloc[0] > 0 else 0
    )

    # One-hot encoding
    df = pd.get_dummies(df, drop_first = True)

    # Align columns with training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    # Scale numeric features
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = scaler.transform(df[num_cols])

    # Prediction
    risk_probability = model.predict_proba(df)[0][1]

    threshold = 0.3
    risk_class = "High Risk" if risk_probability >= threshold else "Low Risk"

    # ----------------------------------
    # Display result
    # ----------------------------------
    st.divider()

    if risk_class == "High Risk":
        st.error("⚠️ High Credit Risk")
    else:
        st.success("✅ Low Credit Risk")

    st.metric(
        label="Risk Probability",
        value=f"{risk_probability:.3f}"
    )

    st.caption(f"Decision Threshold: {threshold}")

