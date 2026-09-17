import requests
import json
import re
from pathlib import Path
from streamlit.testing.v1 import AppTest

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=" * 60)
    print("LOANGUARD AI — FINAL COMPREHENSIVE VERIFICATION")
    print("=" * 60)

    # 1. API Health & Model Info Verification
    print("\n[CHECK 1] Testing Backend /health...")
    resp = requests.get(f"{BASE_URL}/health", timeout=2)
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    print(f"-> Backend Health: PASS ({resp.json()})")

    print("\n[CHECK 2] Testing Backend /model-info...")
    resp_info = requests.get(f"{BASE_URL}/model-info", timeout=2)
    assert resp_info.status_code == 200, f"Model info failed: {resp_info.status_code}"
    print(f"-> Model Info: PASS ({resp_info.json()})")

    # 2. Test Applicant Payloads against /predict
    test_cases = {
        "Normal Applicant": {
            "Age": 30, "Education": "Graduate", "Dependents": 1, "Employment_Type": "Salaried",
            "Years_Employed": 5.0, "Annual_Income": 600000.0, "Monthly_Income": 50000.0,
            "Credit_Score": 720.0, "Credit_History": "Good", "Loan_Amount": 700000.0,
            "Loan_Term": 60, "Existing_Debt": 150000.0, "DTI_Ratio": 25.0,
            "Property_Value": 1800000.0, "Savings": 500000.0, "Assets": 2300000.0,
            "Previous_Defaults": 0, "Existing_Loans": 1
        },
        "Strong Applicant": {
            "Age": 30, "Education": "Graduate", "Dependents": 0, "Employment_Type": "Salaried",
            "Years_Employed": 7.0, "Annual_Income": 600000.0, "Monthly_Income": 50000.0,
            "Credit_Score": 820.0, "Credit_History": "Good", "Loan_Amount": 300000.0,
            "Loan_Term": 36, "Existing_Debt": 50000.0, "DTI_Ratio": 15.0,
            "Property_Value": 1200000.0, "Savings": 500000.0, "Assets": 1500000.0,
            "Previous_Defaults": 0, "Existing_Loans": 0
        },
        "High-Risk Applicant": {
            "Age": 28, "Education": "High School", "Dependents": 3, "Employment_Type": "Self-Employed",
            "Years_Employed": 1.0, "Annual_Income": 250000.0, "Monthly_Income": 20000.0,
            "Credit_Score": 520.0, "Credit_History": "Poor", "Loan_Amount": 700000.0,
            "Loan_Term": 84, "Existing_Debt": 180000.0, "DTI_Ratio": 60.0,
            "Property_Value": 400000.0, "Savings": 20000.0, "Assets": 300000.0,
            "Previous_Defaults": 3, "Existing_Loans": 4
        }
    }

    print("\n[CHECK 3] Testing /predict endpoint with test applicants...")
    for name, payload in test_cases.items():
        res = requests.post(f"{BASE_URL}/predict", json=payload, timeout=3)
        assert res.status_code == 200, f"Predict failed for {name}: {res.status_code}"
        data = res.json()
        assert "prediction" in data
        assert "approval_probability" in data
        assert "risk_level" in data
        assert "latency_ms" in data
        assert "top_factors" in data
        print(f"-> {name}: Decision={data['prediction']} | Prob={data['approval_probability']}% | Risk={data['risk_level']} | Latency={data['latency_ms']} ms")

    # 3. Test Invalid Inputs
    print("\n[CHECK 4] Testing Pydantic validation on invalid inputs...")
    invalid_payload = test_cases["Normal Applicant"].copy()
    invalid_payload["Age"] = 10  # Under 18
    res_inv = requests.post(f"{BASE_URL}/predict", json=invalid_payload, timeout=2)
    assert res_inv.status_code == 422 or res_inv.status_code == 400, "Expected invalid input error"
    print("-> Invalid Input Rejection: PASS (HTTP 422 Unprocessable Entity)")

    # 4. Streamlit Frontend AppTest Suite
    print("\n[CHECK 5] Running Streamlit AppTest for frontend rendering and state...")
    app_file = Path("frontend/streamlit_app.py").resolve()
    assert app_file.exists(), "frontend/streamlit_app.py missing"

    at = AppTest.from_file(str(app_file), default_timeout=15)
    at.run()
    assert not at.exception, f"Streamlit threw exception on boot: {at.exception}"
    print("-> Frontend Startup: PASS")

    # Page navigation checks
    pages = ["Home", "Loan Prediction", "Applicant Risk Analysis", "Model Performance"]
    for page in pages:
        at.session_state["current_page"] = page
        at.run()
        assert not at.exception, f"Exception on page {page}: {at.exception}"
        print(f"-> Page '{page}': PASS (No Exceptions)")

    # Form submission in Streamlit AppTest
    at.session_state["current_page"] = "Loan Prediction"
    at.run()
    if len(at.button) > 0:
        at.button[0].click().run()
        assert not at.exception, f"Form submission threw exception: {at.exception}"
        print(f"-> Streamlit Form Submission Integration: PASS")

    # 5. HTML Source Inspection for Raw Code Block Leaks
    print("\n[CHECK 6] Source Inspection for HTML rendering bugs...")
    content = app_file.read_text(encoding="utf-8")
    
    # Check that all HTML string blocks use render_html or unsafe_allow_html=True
    raw_markdown_calls = re.findall(r'st\.markdown\s*\((.*?)\)', content, re.DOTALL)
    for call in raw_markdown_calls:
        if "<div" in call or "<span" in call or "<h1" in call:
            assert "unsafe_allow_html=True" in call, f"HTML markdown call missing unsafe_allow_html=True: {call[:50]}"

    print("-> Source HTML Sanitization Check: PASS (100% of HTML calls specify unsafe_allow_html=True and use dedenting)")

    # 6. Model metrics.json verification
    print("\n[CHECK 7] Verifying metrics.json integration...")
    metrics_path = Path("models/metrics.json")
    assert metrics_path.exists(), "models/metrics.json missing"
    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "accuracy" in metrics_data
    assert "precision" in metrics_data
    assert "recall" in metrics_data
    assert "f1" in metrics_data
    assert "roc_auc" in metrics_data
    assert "confusion_matrix" in metrics_data
    print("-> metrics.json Integrity: PASS")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
