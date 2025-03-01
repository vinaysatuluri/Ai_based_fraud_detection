from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI()

# Dynamically find and load the model
model_path = os.path.join(os.path.dirname(__file__), "models/final_fraud_detection_model.pkl")
model = joblib.load(model_path)

# Define request body model
class TransactionData(BaseModel):
    features: list  # Expecting a list of transaction features

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is running!"}

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: TransactionData):
    features = np.array(data.features).reshape(1, -1)  # Reshape input
    prediction = model.predict(features)[0]  # Predict fraud (1) or not (0)
    probability = model.predict_proba(features)[0][1]  # Fraud probability score

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(probability * 100, 2)  # Convert to percentage
    }
