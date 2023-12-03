import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, status
import uvicorn
from pydantic import BaseModel

## app
app = FastAPI(
)

# Path of the model
BASE_DIR = Path(__file__).resolve(strict = True).parent

# Load the model
with open(f"{BASE_DIR}/fraud_detection_model-0.0.1.pkl", "rb") as file:
    model = joblib.load(file)

# Data validation
class DataValidation(BaseModel):
    step: int
    type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrg: float
    oldbalanceDest: float
    newbalancedDest: float

# home endpoint 
@app.get("/")
def home():
    return {
        "Message": "Machine learning api to detect credit fraud",
        "Health Check ": "OK",
        "Version": "0.0.1"
    }

# Prediction endpoint 
@app.post("/prediction", status_code = status.HTTP_201_CREATED)
def inference(data : DataValidation):
    # Features dictionary 
    features = data.dict()

    # feature dataframe
    features = pd.DataFrame(features, index = [0])

    # Inference - Predictions
    pred = model.predict(features)
    pred_prob = model.predict_proba(features)

    prob_nofraud = np.round(pred_prob[0, 0]*100, 2)
    prob_fraud = np.round(pred_prob[0, 1]*100, 2)

    if pred == 1:
        return {f"is potentially fraudulent with a probability of {prob_fraud} percent"}
    else:
        return{f"is not potentially fraudulent with a probability of {prob_nofraud} percent"}