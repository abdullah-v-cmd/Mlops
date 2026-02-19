from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# ------------------------
# 1️⃣ Create FastAPI app
# ------------------------
app = FastAPI(title="Telco Churn Prediction API")

# ------------------------
# 2️⃣ Load trained model
# ------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train your model first.")

model = joblib.load(MODEL_PATH)

# ------------------------
# 3️⃣ Define input schema
# ------------------------
class CustomerData(BaseModel):
    customerID: str  # Optional for reference
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

# ------------------------
# 4️⃣ Root endpoint
# ------------------------
@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is running"}

# ------------------------
# 5️⃣ Predict endpoint
# ------------------------
@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert input to dict
        input_dict = data.dict()

        # Remove customerID before prediction
        input_dict.pop("customerID", None)

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Ensure correct types
        input_df = input_df.astype({
            "gender": int,
            "SeniorCitizen": int,
            "Partner": int,
            "Dependents": int,
            "tenure": int,
            "PhoneService": int,
            "MultipleLines": int,
            "InternetService": int,
            "OnlineSecurity": int,
            "OnlineBackup": int,
            "DeviceProtection": int,
            "TechSupport": int,
            "StreamingTV": int,
            "StreamingMovies": int,
            "Contract": int,
            "PaperlessBilling": int,
            "PaymentMethod": int,
            "MonthlyCharges": float,
            "TotalCharges": float
        })

        # Make prediction
        pred = model.predict(input_df)[0]
        result = "Yes" if pred == 1 else "No"

        return {"churn_prediction": result}

    except Exception as e:
        # Return error for debugging
        return {"error": str(e)}
