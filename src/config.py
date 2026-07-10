from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH     = BASE_DIR / "data"  / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH        = BASE_DIR / "models" / "churn_pipeline.joblib"
METRICS_PATH      = BASE_DIR / "models" / "metrics.json"
IMPORTANCES_PATH  = BASE_DIR / "models" / "feature_importances.csv"

TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TEST_SIZE     = 0.2
RANDOM_STATE  = 42
CV_FOLDS      = 5
