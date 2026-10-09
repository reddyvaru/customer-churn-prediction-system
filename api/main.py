from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Locate and load the saved model artifact
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final_churn_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["pipeline"]
THRESHOLD = float(artifact["threshold"])
FEATURE_COLUMNS = artifact["feature_columns"]

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability and risk level.",
    version="1.0.0",
)


# Input validation
class CustomerInput(BaseModel):
    tenure_months: int = Field(ge=0, le=72)
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    cltv: int = Field(ge=0)

    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    senior_citizen: Literal["Yes", "No"]
    gender: Literal["Male", "Female"]

    phone_service: Literal["Yes", "No"]
    multiple_lines: Literal["Yes", "No", "No phone service"]

    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: Literal["Yes", "No", "No internet service"]
    online_backup: Literal["Yes", "No", "No internet service"]
    device_protection: Literal["Yes", "No", "No internet service"]
    tech_support: Literal["Yes", "No", "No internet service"]

    streaming_tv: Literal["Yes", "No", "No internet service"]
    streaming_movies: Literal["Yes", "No", "No internet service"]

    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]

    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


def engineer_features(row: dict) -> pd.DataFrame:
    """Create the two engineered features used during training."""

    service_columns = [
        "Phone Service",
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    ]

    service_count = sum(
        row[col] == "Yes" for col in service_columns
    ) + int(row["Internet Service"] != "No")

    tenure = row["Tenure Months"]

    if tenure <= 12:
        tenure_group = "0-12 months"
    elif tenure <= 24:
        tenure_group = "13-24 months"
    elif tenure <= 48:
        tenure_group = "25-48 months"
    else:
        tenure_group = "49-72 months"

    row["Service Count"] = service_count
    row["Tenure Group"] = tenure_group

    input_df = pd.DataFrame([row])

    # Match the exact column order expected by the trained pipeline.
    return input_df[FEATURE_COLUMNS]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": artifact.get("model_name", "Balanced Logistic Regression"),
        "threshold": THRESHOLD,
    }


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    """Return churn probability, predicted class, and risk level."""

    values = customer.model_dump()

    # Map API field names to the training dataset's column names.
    column_mapping = {
        "tenure_months": "Tenure Months",
        "monthly_charges": "Monthly Charges",
        "total_charges": "Total Charges",
        "cltv": "CLTV",
        "partner": "Partner",
        "dependents": "Dependents",
        "senior_citizen": "Senior Citizen",
        "gender": "Gender",
        "phone_service": "Phone Service",
        "multiple_lines": "Multiple Lines",
        "internet_service": "Internet Service",
        "online_security": "Online Security",
        "online_backup": "Online Backup",
        "device_protection": "Device Protection",
        "tech_support": "Tech Support",
        "streaming_tv": "Streaming TV",
        "streaming_movies": "Streaming Movies",
        "contract": "Contract",
        "paperless_billing": "Paperless Billing",
        "payment_method": "Payment Method",
    }

    row = {
        column_mapping[key]: value
        for key, value in values.items()
    }

    input_df = engineer_features(row)

    churn_probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    predicted_value = int(churn_probability >= THRESHOLD)

    if churn_probability < THRESHOLD:
        risk_level = "Low"
    elif churn_probability < 0.50:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "churn_probability": round(churn_probability, 4),
        "churn_probability_percent": round(churn_probability * 100, 2),
        "predicted_value": predicted_value,
        "predicted_class": (
            "Churn" if predicted_value == 1 else "No Churn"
        ),
        "risk_level": risk_level,
        "threshold_used": THRESHOLD,
    }