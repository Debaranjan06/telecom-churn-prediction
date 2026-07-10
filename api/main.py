import json
from functools import lru_cache
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import MODEL_PATH, METRICS_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES

app = FastAPI(title="Telecom Churn Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schema — strict Literal types enforce valid categorical values
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    gender:              Literal["Male", "Female"]
    SeniorCitizen:       Literal["0", "1"]
    Partner:             Literal["Yes", "No"]
    Dependents:          Literal["Yes", "No"]
    tenure:              int = Field(..., ge=0)
    PhoneService:        Literal["Yes", "No"]
    MultipleLines:       Literal["Yes", "No", "No phone service"]
    InternetService:     Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:      Literal["Yes", "No", "No internet service"]
    OnlineBackup:        Literal["Yes", "No", "No internet service"]
    DeviceProtection:    Literal["Yes", "No", "No internet service"]
    TechSupport:         Literal["Yes", "No", "No internet service"]
    StreamingTV:         Literal["Yes", "No", "No internet service"]
    StreamingMovies:     Literal["Yes", "No", "No internet service"]
    Contract:            Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling:    Literal["Yes", "No"]
    PaymentMethod:       Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges:      float = Field(..., ge=0)
    TotalCharges:        float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Model loading — lazy, cached, fails gracefully
# ---------------------------------------------------------------------------

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model not found at {MODEL_PATH}. "
                    "Run `python -m src.train` first to train and save the pipeline."
                ),
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="metrics.json not found. Run training first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/predict")
def predict(customer: CustomerFeatures):
    pipeline = get_pipeline()

    col_order = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    row = customer.model_dump()
    df = pd.DataFrame([row])[col_order]

    prob = float(pipeline.predict_proba(df)[0, 1])
    prediction = "Yes" if prob >= 0.5 else "No"

    if prob < 0.4:
        risk_band = "low"
    elif prob < 0.7:
        risk_band = "medium"
    else:
        risk_band = "high"

    return {
        "churn_probability": round(prob, 4),
        "prediction": prediction,
        "risk_band": risk_band,
    }
