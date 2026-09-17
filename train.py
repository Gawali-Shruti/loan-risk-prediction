from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Loan_Status"

NUMERIC_FEATURES = [
    "Age", "Dependents", "Years_Employed", "Annual_Income",
    "Monthly_Income", "Credit_Score", "Loan_Amount", "Loan_Term",
    "Existing_Debt", "DTI_Ratio", "Property_Value", "Savings",
    "Assets", "Previous_Defaults", "Existing_Loans"
]

CATEGORICAL_FEATURES = [
    "Education", "Employment_Type", "Credit_History"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES)
    ])


def main():
    df = pd.read_csv(DATA_PATH)

    # Basic validation
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.drop_duplicates().copy()

    # Target validation
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Loan_Status must contain only 0 (Rejected) and 1 (Approved).")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Fit preprocessing ONLY on training data.
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # SMOTE is applied only to training data.
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_t, y_train)

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test_t)
    y_prob = model.predict_proba(X_test_t)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "model_name": "RandomForestClassifier",
        "model_version": "1.0.0",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Save separate artifacts, matching the project specification.
    joblib.dump(model, MODEL_DIR / "loan_model.pkl")
    joblib.dump(preprocessor.named_transformers_["num"].named_steps["scaler"],
                MODEL_DIR / "scaler.pkl")
    joblib.dump(preprocessor.named_transformers_["cat"].named_steps["encoder"],
                MODEL_DIR / "encoder.pkl")
    joblib.dump(ALL_FEATURES, MODEL_DIR / "feature_columns.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(MODEL_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print("\nTraining complete.")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
