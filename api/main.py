from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# paths
MODEL_PATH = "models/model.pkl"

# load model
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Telco Churn Prediction API")

# define input schema
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    # convert to dataframe
    input_df = pd.DataFrame([data.dict()])

    # make prediction
    pred = model.predict(input_df)[0]

    # convert numeric to Yes/No
    result = "Yes" if pred == 1 else "No"

    return {"churn_prediction": result}
