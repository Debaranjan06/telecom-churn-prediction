import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from src.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    MODEL_PATH, METRICS_PATH, IMPORTANCES_PATH,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS,
)
from src.data_preprocessing import load_and_clean


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
    ])


def build_pipelines(scale_pos_weight):
    preprocessor = build_preprocessor()
    return {
        "LogisticRegression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", RandomForestClassifier(
                class_weight="balanced", n_estimators=400,
                min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", XGBClassifier(
                scale_pos_weight=scale_pos_weight, n_estimators=400,
                learning_rate=0.05, max_depth=4,
                random_state=RANDOM_STATE, eval_metric="logloss",
                verbosity=0,
            )),
        ]),
    }


def get_feature_names(pipeline):
    ct = pipeline.named_steps["preprocess"]
    return ct.get_feature_names_out()


def get_importances(pipeline, feature_names):
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    # LogisticRegression — use absolute coefficients
    return np.abs(model.coef_[0])


def main():
    print("=" * 60)
    print("Loading and cleaning data...")
    X, y = load_and_clean()
    print(f"  Samples: {len(X)}  |  Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"  Class ratio (neg/pos): {scale_pos_weight:.2f}")

    pipelines = build_pipelines(scale_pos_weight)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("\n" + "=" * 60)
    print(f"Running {CV_FOLDS}-fold stratified CV on training split...")
    cv_results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}
        print(f"  {name:20s}  AUC = {scores.mean():.4f} +/- {scores.std():.4f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
    print(f"\n  Best model: {best_name}  (CV AUC = {cv_results[best_name]['mean']:.4f})")

    print("\n" + "=" * 60)
    print(f"Refitting {best_name} on full training split...")
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(X_train, y_train)

    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = best_pipeline.predict(X_test)

    test_metrics = {
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "f1":        float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall":    float(recall_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\nTest-set results:")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:15s}: {v:.4f}")
    print(f"  confusion_matrix: {test_metrics['confusion_matrix']}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Save artifacts
    print("=" * 60)
    print("Saving artifacts...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"  Pipeline    -> {MODEL_PATH}")

    metrics_payload = {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  Metrics     -> {METRICS_PATH}")

    feature_names = get_feature_names(best_pipeline)
    importances   = get_importances(best_pipeline, feature_names)
    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(IMPORTANCES_PATH, index=False)
    print(f"  Importances -> {IMPORTANCES_PATH}")
    print("\nTop 10 features:")
    print(imp_df.head(10).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
