# LoanGuard AI

> **AI-Powered Loan Risk Prediction Platform**

LoanGuard AI is an end-to-end machine learning platform engineered for credit risk assessment, loan eligibility prediction, and financial factor analysis. Built with a **FastAPI** backend and an interactive **Streamlit** dashboard, the system processes applicant financial profiles in real time to deliver risk classifications, approval probabilities, and model factor importances.

---

## Overview

LoanGuard AI assists financial institutions and underwriters by evaluating loan applications using trained machine learning models. By entering applicant details—such as income, credit score, debt-to-income (DTI) ratio, existing debt, and asset history—the platform calculates the probability of loan approval, categorizes risk into actionable bands (Low, Medium, High), and surfaces the primary drivers behind each decision.

---

## Features

- **AI Loan Eligibility Prediction**: Instant approval/rejection predictions powered by a tuned Random Forest Classifier.
- **Credit Risk Assessment**: Quantitative evaluation of financial risk based on credit score, defaults, and debt ratio.
- **Approval Probability**: Precise approval likelihood score (0–100%).
- **Risk Classification**: Automated risk categorizations (`Low`, `Medium`, `High`) based on probability thresholds.
- **Model Factor Analysis**: Identification of top features influencing the prediction for each individual applicant.
- **Applicant Financial Analysis**: Dynamic visualizations comparing applicant metrics against benchmark distributions.
- **Model Performance Dashboard**: Real-time inspection of accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.
- **FastAPI Backend**: High-performance RESTful API with Pydantic request validation and structured exception handling.
- **Streamlit Frontend**: Responsive multi-page web dashboard with dark UI styling and interactive Plotly visualization.

---

## Architecture

```
Streamlit Frontend
        ↓
 FastAPI Backend
        ↓
   Preprocessor
        ↓
Random Forest Model
        ↓
Prediction Response
        ↓
Streamlit Dashboard
```

1. **Streamlit Frontend**: User enters applicant details or uploads data for analysis.
2. **FastAPI Backend**: Validates incoming request body via Pydantic schemas.
3. **Preprocessor**: Imputes missing values, scales numerical features via `StandardScaler`, and encodes categorical fields via `OneHotEncoder`.
4. **Random Forest Model**: Predicts approval probability using 350 decision trees trained on balanced datasets.
5. **Prediction Response**: Returns decision, risk level, probability, latency, and top feature importances to the dashboard.

---

## Project Structure

```
loan-risk-prediction/
│
├── backend/
│   ├── __init__.py
│   ├── app.py                  # FastAPI REST endpoints (/health, /model-info, /predict)
│   └── predictor.py            # Model loading, preprocessing, and inference logic
├── data/
│   └── loan_dataset.csv        # Historical training dataset (1,500 rows)
├── frontend/
│   └── streamlit_app.py        # Streamlit multi-page web frontend application
├── models/
│   ├── encoder.pkl             # Categorical OneHotEncoder artifact
│   ├── feature_columns.pkl     # Input feature column mapping
│   ├── feature_names.json      # Encoded feature names schema
│   ├── loan_model.pkl          # Trained RandomForestClassifier model artifact (7.47 MB)
│   ├── metrics.json            # Model evaluation performance metrics
│   ├── preprocessor.pkl        # Scikit-learn ColumnTransformer pipeline artifact
│   └── scaler.pkl              # Numerical StandardScaler artifact
├── utils/
│   ├── __init__.py
│   ├── validation.py           # Dataset schema validation helpers
│   └── visualization.py        # Matplotlib visualization utility routines
├── test_api.py                 # Endpoint smoke test script
├── test_final_verification.py  # Comprehensive end-to-end test suite (API + Streamlit AppTest)
├── test_prediction_validation_suite.py  # Multi-applicant end-to-end validation suite
├── train.py                    # Model training, preprocessing, SMOTE, and artifact exporter
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git exclusion rules
```

---

## Installation

### Prerequisites
- Python 3.10+ installed on your system.

### 1. Clone the repository
```bash
git clone <repository-url>
cd loan-risk-prediction
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run Backend

Launch the FastAPI application server using Uvicorn:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`. Interactive API documentation (Swagger UI) can be accessed at `http://127.0.0.1:8000/docs`.

---

## Run Frontend

In a separate terminal window (with the virtual environment activated), start the Streamlit application:

```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

Access the user interface in your web browser at `http://localhost:8501`.

---

## API Documentation

### 1. Health Check
- **Endpoint**: `GET /health`
- **Description**: Verifies backend service status.
- **Response Example**:
```json
{
  "status": "ok",
  "service": "loan-risk-api"
}
```

### 2. Model Information
- **Endpoint**: `GET /model-info`
- **Description**: Returns model metadata and cross-validation evaluation metrics.
- **Response Example**:
```json
{
  "model_name": "RandomForestClassifier",
  "model_version": "1.0.0",
  "accuracy": 0.8,
  "precision": 0.8523,
  "recall": 0.7697,
  "f1": 0.8089,
  "roc_auc": 0.8886
}
```

### 3. Predict Loan Risk
- **Endpoint**: `POST /predict`
- **Description**: Evaluates loan applicant financial data and returns loan decision with risk scores.

#### Request Body Schema (`Applicant`):
```json
{
  "Age": 30,
  "Education": "Graduate",
  "Dependents": 0,
  "Employment_Type": "Salaried",
  "Years_Employed": 5.0,
  "Annual_Income": 600000.0,
  "Monthly_Income": 50000.0,
  "Credit_Score": 720.0,
  "Credit_History": "Good",
  "Loan_Amount": 700000.0,
  "Loan_Term": 60,
  "Existing_Debt": 150000.0,
  "DTI_Ratio": 25.0,
  "Property_Value": 1800000.0,
  "Savings": 500000.0,
  "Assets": 2300000.0,
  "Previous_Defaults": 0,
  "Existing_Loans": 1
}
```

#### Response Example:
```json
{
  "prediction": "Approved",
  "approval_probability": 84.52,
  "risk_level": "Low",
  "top_factors": [
    { "feature": "num__Credit_Score", "importance": 0.18432 },
    { "feature": "num__DTI_Ratio", "importance": 0.15211 },
    { "feature": "num__Annual_Income", "importance": 0.12845 }
  ],
  "latency_ms": 12.45
}
```

---

## Machine Learning

The core risk engine uses an ensemble **RandomForestClassifier** trained on applicant financial data.

- **Preprocessing Pipeline**:
  - **Numerical Features**: Missing value median imputation followed by `StandardScaler` normalization.
  - **Categorical Features**: Missing value mode imputation followed by `OneHotEncoder` transformation.
  - **Class Imbalance Handling**: Synthetic Minority Over-sampling Technique (`SMOTE`) applied exclusively to the training split.
- **Model Hyperparameters**:
  - `n_estimators`: 350
  - `max_depth`: 12
  - `min_samples_leaf`: 3
  - `class_weight`: `"balanced"`
- **Training Pipeline Script**: `train.py` can be executed to retrain the model and regenerate model artifacts.

---

## Model Metrics

Evaluation results recorded on the test set (`models/metrics.json`):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 80.00% |
| **Precision** | 85.23% |
| **Recall** | 76.97% |
| **F1-Score** | 80.89% |
| **ROC-AUC** | 88.86% |
| **Training Samples** | 1,200 rows |
| **Test Samples** | 300 rows |

### Confusion Matrix
```
               Predicted Rejected    Predicted Approved
Actual Rejected      113 (TN)               22 (FP)
Actual Approved       38 (FN)              127 (TP)
```

---

## Testing

The project includes unit, validation, and integration tests:

1. **API Smoke Test**:
   ```bash
   python test_api.py
   ```
2. **End-to-End Validation Suite**:
   ```bash
   python test_prediction_validation_suite.py
   ```
3. **Comprehensive Verification (API + Streamlit AppTest)**:
   ```bash
   python test_final_verification.py
   ```

---

## Screenshots

*(UI screenshots can be captured from the running Streamlit dashboard at `http://localhost:8501` and added to project documentation.)*

---

## Disclaimer

This application is created for educational, research, and decision-support demonstration purposes. Predictions generated by this system are produced by machine learning models trained on synthetic data and must not be construed as financial advice or binding lending decisions.
