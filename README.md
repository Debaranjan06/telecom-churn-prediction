# Telecom Churn Prediction

End-to-end machine learning pipeline that predicts customer churn for a telecommunications provider, built on the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                               │
│  data/WA_Fn-UseC_-Telco-Customer-Churn.csv  (7,043 customers)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  src/data_preprocessing.py
                            │  (stateless raw cleanup only)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                           │
│  src/train.py                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  sklearn Pipeline                                        │   │
│  │  ┌─────────────────────┐   ┌────────────────────────┐   │   │
│  │  │  ColumnTransformer  │──▶│  Best Model (auto-sel) │   │   │
│  │  │  StandardScaler     │   │  LogisticRegression /  │   │   │
│  │  │  OneHotEncoder      │   │  RandomForest /        │   │   │
│  │  └─────────────────────┘   │  XGBoost               │   │   │
│  │                            └────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Artifacts: models/churn_pipeline.joblib                        │
│             models/metrics.json                                  │
│             models/feature_importances.csv                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              ▼                            ▼
┌─────────────────────────┐  ┌────────────────────────────────────┐
│   FastAPI Service       │  │   Streamlit Demo                   │
│   api/main.py           │  │   app/streamlit_app.py             │
│   POST /predict         │  │   (loads pipeline directly)        │
│   GET  /health          │  │   Three-column form                │
│   GET  /model-info      │  │   Risk band + feature chart        │
└──────────┬──────────────┘  └────────────────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│   React + Vite Frontend │
│   frontend/src/         │
│   Proxies /api/* to     │
│   FastAPI :8000         │
└─────────────────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **All transformations inside the sklearn Pipeline** | Preprocessing (scaling, encoding) is baked into `churn_pipeline.joblib`. The same object is used at training and serving time — no train/serve skew possible. |
| **Class imbalance via class weights** | `class_weight="balanced"` (LR, RF) and `scale_pos_weight=neg/pos` (XGBoost) — no resampling, no data leakage risk. |
| **Model selection via stratified CV** | 5-fold stratified cross-validation on the training split only. The test set is touched exactly once for the final report. |
| **Pydantic `Literal` types on every field** | Every valid categorical value is encoded as a type — invalid input returns HTTP 422 before it reaches the model. |
| **`churn_pipeline.joblib` committed to Git** | Streamlit Community Cloud runs `streamlit run app/streamlit_app.py` with no build step, so the trained artifact must be in the repo. The raw CSV is excluded (too large). |

---

## Results

| Model | CV AUC (5-fold) | Test AUC | Test F1 | Test Recall |
|---|---|---|---|---|
| **Logistic Regression** ✓ | **0.8460** | **0.8415** | **0.614** | **0.783** |
| Random Forest | 0.8449 | — | — | — |
| XGBoost | 0.8409 | — | — | — |

Confusion matrix (test set, 1,409 samples):

```
               Predicted No    Predicted Yes
Actual No           747              288
Actual Yes           81              293
```

---

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ (for the React frontend)

### Install

```bash
git clone <your-repo-url>
cd telecom-churn-prediction

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### Download the dataset

Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/`.

### Train

```bash
python -m src.train
```

Saves three artifacts to `models/`.

---

## Running

### FastAPI backend

```bash
uvicorn api.main:app --reload
# API docs → http://localhost:8000/docs
```

### React frontend (requires FastAPI running)

```bash
cd frontend
npm run dev
# → http://localhost:3000
```

### Streamlit demo (standalone — no API needed)

```bash
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

---

## API Usage

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0
  }'
```

Response:

```json
{
  "churn_probability": 0.8951,
  "prediction": "Yes",
  "risk_band": "high"
}
```

### GET /health

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

### GET /model-info

```bash
curl http://localhost:8000/model-info
# Returns contents of models/metrics.json
```

---

## Project Structure

```
telecom-churn-prediction/
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, feature lists, constants
│   ├── data_preprocessing.py   # Stateless raw cleanup
│   └── train.py                # Training pipeline + model selection
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI inference service
│   └── test_api.py             # TestClient tests
├── app/
│   └── streamlit_app.py        # Streamlit demo
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── PredictionForm.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   └── FeatureImportanceChart.jsx
│   ├── vite.config.js
│   └── package.json
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis
├── models/
│   ├── churn_pipeline.joblib   # Trained pipeline (committed)
│   ├── metrics.json
│   └── feature_importances.csv
├── data/                       # CSV goes here (git-ignored)
├── requirements.txt
└── .gitignore
```

---

## Deployment

### Streamlit Community Cloud

1. Push this repo to GitHub (ensure `models/churn_pipeline.joblib` is committed).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set:
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.
5. Your app is live at `https://<your-app>.streamlit.app`.

> The app loads `churn_pipeline.joblib` directly — no FastAPI server needed on the cloud.

### Render (FastAPI backend)

1. Push the repo to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo.
3. Set:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Click **Deploy**.

### React Frontend (Vercel / Netlify)

1. Update `vite.config.js` proxy target to your Render URL.
2. Run `npm run build` — deploy the `frontend/dist/` folder to Vercel or Netlify.
