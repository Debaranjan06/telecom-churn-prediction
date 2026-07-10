import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}

LOYAL = {
    "gender": "Male", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
}

INVALID = {
    "gender": "Attack Helicopter",   # not a valid Literal
    "tenure": -5,                     # below ge=0
    "MonthlyCharges": -10.0,
}


def test_high_risk():
    r = client.post("/predict", json=HIGH_RISK)
    print(f"\n[HIGH-RISK]  status={r.status_code}  body={r.json()}")
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ("Yes", "No")
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_band"] in ("low", "medium", "high")


def test_loyal():
    r = client.post("/predict", json=LOYAL)
    print(f"[LOYAL]      status={r.status_code}  body={r.json()}")
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ("Yes", "No")


def test_invalid():
    r = client.post("/predict", json=INVALID)
    print(f"[INVALID]    status={r.status_code}  body={r.json()}")
    assert r.status_code == 422


if __name__ == "__main__":
    test_high_risk()
    test_loyal()
    test_invalid()
    print("\nAll 3 tests passed.")
