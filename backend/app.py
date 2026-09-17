from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from backend.predictor import predict_applicant, metrics

app = FastAPI(
    title="Intelligent Loan Approval & Credit Risk API",
    version="1.0.0"
)


class Applicant(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Education: str
    Dependents: int = Field(..., ge=0, le=20)
    Employment_Type: str
    Years_Employed: float = Field(..., ge=0, le=80)
    Annual_Income: float = Field(..., gt=0)
    Monthly_Income: float = Field(..., gt=0)
    Credit_Score: float = Field(..., ge=300, le=900)
    Credit_History: str
    Loan_Amount: float = Field(..., gt=0)
    Loan_Term: int = Field(..., ge=1, le=120)
    Existing_Debt: float = Field(..., ge=0)
    DTI_Ratio: float = Field(..., ge=0, le=100)
    Property_Value: float = Field(..., ge=0)
    Savings: float = Field(..., ge=0)
    Assets: float = Field(..., ge=0)
    Previous_Defaults: int = Field(..., ge=0, le=50)
    Existing_Loans: int = Field(..., ge=0, le=50)


@app.get("/health")
def health():
    return {"status": "ok", "service": "loan-risk-api"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": metrics["model_name"],
        "model_version": metrics["model_version"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"]
    }


@app.post("/predict")
def predict(applicant: Applicant):
    try:
        return predict_applicant(applicant.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
