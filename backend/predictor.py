from pathlib import Path
import json
import time
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

model = joblib.load(MODEL_DIR / "loan_model.pkl")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

with open(MODEL_DIR / "metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)


def risk_level(approval_probability: float) -> str:
    # Business/demo thresholds. Calibrate these with domain experts for production.
    if approval_probability >= 0.70:
        return "Low"
    if approval_probability >= 0.40:
        return "Medium"
    return "High"


def predict_applicant(payload: dict) -> dict:
    start = time.perf_counter()

    row = {k: payload[k] for k in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)

    X = preprocessor.transform(df)
    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= 0.50)

    # Global feature importance for this applicant.
    transformed_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    ranked = sorted(
        zip(transformed_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:8]

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": round(probability * 100, 2),
        "risk_level": risk_level(probability),
        "top_factors": [
            {"feature": str(name), "importance": round(float(value), 5)}
            for name, value in ranked
        ],
        "latency_ms": round(latency_ms, 3)
    }
