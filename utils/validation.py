REQUIRED_COLUMNS = [
    "Age", "Education", "Dependents", "Employment_Type",
    "Years_Employed", "Annual_Income", "Monthly_Income",
    "Credit_Score", "Credit_History", "Loan_Amount", "Loan_Term",
    "Existing_Debt", "DTI_Ratio", "Property_Value", "Savings",
    "Assets", "Previous_Defaults", "Existing_Loans", "Loan_Status"
]


def validate_dataset(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"

    if df["Loan_Status"].dropna().isin([0, 1]).all() is False:
        return False, "Loan_Status must contain only 0 and 1."

    return True, "Dataset validation passed."
