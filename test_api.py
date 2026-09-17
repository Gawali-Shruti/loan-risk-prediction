import requests

payload = {
    "Age": 30,
    "Education": "Graduate",
    "Dependents": 0,
    "Employment_Type": "Salaried",
    "Years_Employed": 5,
    "Annual_Income": 600000,
    "Monthly_Income": 50000,
    "Credit_Score": 720,
    "Credit_History": "Good",
    "Loan_Amount": 700000,
    "Loan_Term": 60,
    "Existing_Debt": 150000,
    "DTI_Ratio": 25,
    "Property_Value": 1800000,
    "Savings": 500000,
    "Assets": 2300000,
    "Previous_Defaults": 0,
    "Existing_Loans": 1
}

r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
print(r.status_code)
print(r.json())
