import sys
from pathlib import Path

# Make src/ importable when run from project root: streamlit run app/streamlit_app.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_PATH, IMPORTANCES_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Model loading — cached so it's loaded once per session
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_importances():
    if not IMPORTANCES_PATH.exists():
        return None
    return pd.read_csv(IMPORTANCES_PATH)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📡 Telecom Churn Predictor")
st.markdown("End-to-end ML pipeline on the **Kaggle Telco Customer Churn** dataset.")

pipeline = load_pipeline()
if pipeline is None:
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Run `python -m src.train` from the project root first.",
        icon="🚨",
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Input form — three columns
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Profile")
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior          = st.selectbox("Senior Citizen", ["0", "1"], format_func=lambda x: "Yes" if x == "1" else "No")
    partner         = st.selectbox("Partner", ["Yes", "No"])
    dependents      = st.selectbox("Dependents", ["Yes", "No"])
    tenure          = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

with col2:
    st.subheader("Services")
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec      = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_bkp      = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot     = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support    = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_mov   = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("Billing")
    contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment         = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=1.0)

st.divider()

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
if st.button("Predict Churn", type="primary", use_container_width=True):
    row = {
        "tenure":          tenure,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    total_charges,
        "gender":          gender,
        "SeniorCitizen":   senior,
        "Partner":         partner,
        "Dependents":      dependents,
        "PhoneService":    phone_service,
        "MultipleLines":   multiple_lines,
        "InternetService": internet,
        "OnlineSecurity":  online_sec,
        "OnlineBackup":    online_bkp,
        "DeviceProtection":device_prot,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_mov,
        "Contract":        contract,
        "PaperlessBilling":paperless,
        "PaymentMethod":   payment,
    }

    col_order = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = pd.DataFrame([row])[col_order]
    prob = float(pipeline.predict_proba(df)[0, 1])
    prediction = "Yes" if prob >= 0.5 else "No"

    if prob < 0.4:
        risk_band, band_emoji, band_color = "Low Risk",    "🟢", "normal"
    elif prob < 0.7:
        risk_band, band_emoji, band_color = "Medium Risk", "🟡", "off"
    else:
        risk_band, band_emoji, band_color = "High Risk",   "🔴", "inverse"

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{prob:.1%}")
    m2.metric("Prediction", f"Churn: {prediction}")
    m3.metric("Risk Band", f"{band_emoji} {risk_band}")

    # Progress bar
    st.progress(prob, text=f"Churn probability: {prob:.1%}")

    if prob >= 0.7:
        st.error("High churn risk — immediate retention action recommended.", icon="🔴")
    elif prob >= 0.4:
        st.warning("Moderate churn risk — consider a proactive retention offer.", icon="🟡")
    else:
        st.success("Low churn risk — focus on upsell opportunities.", icon="🟢")

    # Feature importance expander
    imp_df = load_importances()
    if imp_df is not None:
        with st.expander("Top 10 Feature Importances", expanded=True):
            top10 = imp_df.head(10).copy()
            top10["feature"] = top10["feature"].str.replace(r"^(num|cat)__", "", regex=True)
            st.bar_chart(top10.set_index("feature")["importance"])
