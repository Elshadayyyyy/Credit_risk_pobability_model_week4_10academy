import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

mlflow.set_tracking_uri(
    "sqlite:///D:/gigggs/10Academy/week4-10academy/notebooks/mlflow.db"
)

app = FastAPI(
    title="Credit Risk API",
    version="1.0"
)

model = mlflow.pyfunc.load_model(
    "models:/CreditRiskModel/Production"
)

class CreditRiskInput(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: float
    std_amount: float
    avg_transaction_hour: float
    avg_transaction_day: float
    avg_transaction_month: float


@app.post("/predict")
def predict(data: CreditRiskInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)

    return {
        "is_high_risk": int(prediction[0])
    }
