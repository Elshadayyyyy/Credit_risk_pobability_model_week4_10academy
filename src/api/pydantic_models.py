from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    recency: int
    frequency: int
    monetary: float
    # add other features as needed

class PredictionResponse(BaseModel):
    risk_probability: float
