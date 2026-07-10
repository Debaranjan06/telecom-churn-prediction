import pandas as pd
from src.config import RAW_DATA_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET


def load_and_clean(path=RAW_DATA_PATH):
    """
    Stateless raw cleanup only — no scaling, encoding, or fitting happens here.
    All learned transformations live in the sklearn Pipeline (see train.py).
    """
    df = pd.read_csv(path)

    # TotalCharges is stored as a string; blank values appear for tenure=0 customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # SeniorCitizen comes as int (0/1) — treat as categorical string to match schema
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # customerID carries no signal
    df = df.drop(columns=["customerID"])

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = (df[TARGET] == "Yes").astype(int)

    return X, y
