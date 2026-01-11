from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "credit_risk_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

app = FastAPI(title = "Credit Risk Prediction API")

# Model
class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: int
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: int
    Property_Area: str
    Total_Income: float

# Endpoint - predict
@app.post("/predict")
def predict_risk(data: LoanApplication):

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Feature engineering
    df["Loan_Income_Ratio"] = df["LoanAmount"] / df["Total_Income"]

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training data
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]  # correct order

    # Scale numeric columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = scaler.transform(df[num_cols])

    # Predict
    risk_prob = model.predict_proba(df)[0][1]

    threshold = 0.3
    risk_class = "High Risk" if risk_prob >= threshold else "Low Risk"

    return {
        "Risk_Probability": round(float(risk_prob), 3),
        "Risk_Class": risk_class,
        "Threshold": threshold
    }



