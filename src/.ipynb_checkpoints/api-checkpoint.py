{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "902fd899-9f55-45f5-9961-034c58f740f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI\n",
    "import joblib\n",
    "import numpy as np\n",
    "from pydantic import BaseModel\n",
    "import os\n",
    "\n",
    "# Initialize FastAPI app\n",
    "app = FastAPI()\n",
    "\n",
    "# Dynamically find and load the model\n",
    "model_path = os.path.join(os.path.dirname(__file__), \"models/final_fraud_detection_model.pkl\")\n",
    "model = joblib.load(model_path)\n",
    "\n",
    "# Define request body model\n",
    "class TransactionData(BaseModel):\n",
    "    features: list  # Expecting a list of transaction features\n",
    "\n",
    "# Define the root endpoint\n",
    "@app.get(\"/\")\n",
    "def read_root():\n",
    "    return {\"message\": \"Fraud Detection API is running!\"}\n",
    "\n",
    "# Define the prediction endpoint\n",
    "@app.post(\"/predict\")\n",
    "async def predict(data: TransactionData):\n",
    "    features = np.array(data.features).reshape(1, -1)  # Reshape input\n",
    "    prediction = model.predict(features)[0]  # Predict fraud (1) or not (0)\n",
    "    probability = model.predict_proba(features)[0][1]  # Fraud probability score\n",
    "\n",
    "    return {\n",
    "        \"fraud_prediction\": int(prediction),\n",
    "        \"fraud_probability\": round(probability * 100, 2)  # Convert to percentage\n",
    "    }\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
