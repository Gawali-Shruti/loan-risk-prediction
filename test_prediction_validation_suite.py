import sys
import requests
from streamlit.testing.v1 import AppTest

sys.stdout.reconfigure(encoding='utf-8')

API_URL = "http://127.0.0.1:8000"

def run_suite():
    print("==================================================")
    print("LOANGUARD AI — END-TO-END VALIDATION & TESTING")
    print("==================================================\n")

    # ----------------------------------------------------
    # STEP 1: VERIFY BACKEND CONNECTION
    # ----------------------------------------------------
    print("STEP 1 — Verifying Backend Connection...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        assert r.status_code == 200, f"Health check failed with status {r.status_code}"
        health_data = r.json()
        print(f"[PASS] Backend Health Check: HTTP {r.status_code} -> {health_data}")
    except Exception as exc:
        print(f"[FAIL] Backend connection error: {exc}")
        sys.exit(1)

    # ----------------------------------------------------
    # STEP 2: TEST REALISTIC APPLICANTS
    # ----------------------------------------------------
    print("\nSTEP 2 — Testing 5 Realistic Applicants against live ML model...")

    test_cases = [
        ("Test Case 1 — Strong Applicant", {
            "Age": 30, "Education": "Graduate", "Dependents": 0, "Employment_Type": "Salaried", "Years_Employed": 7.0,
            "Annual_Income": 600000.0, "Monthly_Income": 50000.0, "Existing_Debt": 50000.0, "DTI_Ratio": 15.0, "Savings": 500000.0,
            "Credit_Score": 820.0, "Credit_History": "Good", "Loan_Amount": 300000.0, "Loan_Term": 36, "Property_Value": 1200000.0,
            "Assets": 1500000.0, "Previous_Defaults": 0, "Existing_Loans": 0
        }),
        ("Test Case 2 — Moderate Applicant", {
            "Age": 35, "Education": "Graduate", "Dependents": 2, "Employment_Type": "Salaried", "Years_Employed": 5.0,
            "Annual_Income": 500000.0, "Monthly_Income": 40000.0, "Existing_Debt": 120000.0, "DTI_Ratio": 30.0, "Savings": 200000.0,
            "Credit_Score": 680.0, "Credit_History": "Fair", "Loan_Amount": 500000.0, "Loan_Term": 60, "Property_Value": 800000.0,
            "Assets": 900000.0, "Previous_Defaults": 1, "Existing_Loans": 1
        }),
        ("Test Case 3 — High-Risk Applicant", {
            "Age": 28, "Education": "High School", "Dependents": 3, "Employment_Type": "Self-Employed", "Years_Employed": 1.0,
            "Annual_Income": 250000.0, "Monthly_Income": 20000.0, "Existing_Debt": 180000.0, "DTI_Ratio": 60.0, "Savings": 20000.0,
            "Credit_Score": 520.0, "Credit_History": "Poor", "Loan_Amount": 700000.0, "Loan_Term": 84, "Property_Value": 400000.0,
            "Assets": 300000.0, "Previous_Defaults": 3, "Existing_Loans": 4
        }),
        ("Test Case 4 — Excellent Credit Profile", {
            "Age": 42, "Education": "Graduate", "Dependents": 1, "Employment_Type": "Salaried", "Years_Employed": 15.0,
            "Annual_Income": 1200000.0, "Monthly_Income": 100000.0, "Existing_Debt": 100000.0, "DTI_Ratio": 10.0, "Savings": 1000000.0,
            "Credit_Score": 880.0, "Credit_History": "Good", "Loan_Amount": 400000.0, "Loan_Term": 36, "Property_Value": 2500000.0,
            "Assets": 3000000.0, "Previous_Defaults": 0, "Existing_Loans": 0
        }),
        ("Test Case 5 — Heavy Debt / Low Credit", {
            "Age": 45, "Education": "High School", "Dependents": 2, "Employment_Type": "Self-Employed", "Years_Employed": 2.0,
            "Annual_Income": 300000.0, "Monthly_Income": 25000.0, "Existing_Debt": 250000.0, "DTI_Ratio": 70.0, "Savings": 10000.0,
            "Credit_Score": 450.0, "Credit_History": "Poor", "Loan_Amount": 800000.0, "Loan_Term": 84, "Property_Value": 500000.0,
            "Assets": 350000.0, "Previous_Defaults": 4, "Existing_Loans": 5
        }),
    ]

    responses = {}
    for name, payload in test_cases:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        assert resp.status_code == 200, f"Predict failed for {name}"
        data = resp.json()
        responses[name] = data
        print(f"  [{name}]")
        print(f"    Decision: {data.get('prediction')} | Prob: {data.get('approval_probability')}% | Risk: {data.get('risk_level')} | Latency: {data.get('latency_ms')} ms")
        print(f"    Top Factor #1: {data.get('top_factors')[0] if data.get('top_factors') else 'None'}")

    # ----------------------------------------------------
    # STEP 3: STREAMLIT APPTEST INTEGRATION
    # ----------------------------------------------------
    print("\nSTEP 3 — Verifying Streamlit AppTest End-to-End Integration...")
    at = AppTest.from_file("frontend/streamlit_app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App startup exception: {[e.value for e in at.exception]}"
    print("[PASS] Streamlit frontend boots without exceptions.")

    # Test Prediction Page Submission
    at.session_state["current_page"] = "Loan Prediction"
    at.run()

    pred_btn = None
    for b in at.button:
        if "Predict Loan Eligibility" in b.label:
            pred_btn = b
            break
    assert pred_btn is not None, "Predict Loan Eligibility button not found!"
    pred_btn.click()
    at.run()
    assert not at.exception, f"Form submission exception: {[e.value for e in at.exception]}"
    assert "last_prediction" in at.session_state and at.session_state["last_prediction"] is not None, "Last prediction state missing!"
    print("[PASS] Form submission executed against live FastAPI model.")

    # ----------------------------------------------------
    # STEP 4: OFFLINE API HANDLING
    # ----------------------------------------------------
    print("\nSTEP 4 — Testing API Offline Handling...")
    at.session_state["api_url"] = "http://127.0.0.1:9999"
    at.run()
    assert not at.exception, "App crashed when API URL points to offline port!"
    print("[PASS] Streamlit app handles offline API URL gracefully without crashing.")
    at.session_state["api_url"] = API_URL
    at.run()

    # ----------------------------------------------------
    # STEP 5: HTML LEAK AUDIT
    # ----------------------------------------------------
    print("\nSTEP 5 — Auditing HTML Rendering across all 4 pages...")
    pages = ["Home", "Loan Prediction", "Applicant Risk Analysis", "Model Performance"]
    for p in pages:
        at.session_state["current_page"] = p
        at.run()
        assert not at.exception, f"Exception on page {p}"
        for m in at.markdown:
            val = m.value
            if "```" in val and ("<div" in val or "<span" in val or "<h1" in val or "<p" in val):
                raise AssertionError(f"HTML leak detected on page {p}: {val}")
    print("[PASS] Zero raw HTML leaks detected across all pages.")

    print("\n==================================================")
    print("ALL VALIDATION TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_suite()
