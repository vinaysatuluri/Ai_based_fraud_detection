{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6b14a184-1b3c-41f9-94bc-b819662d754b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Authtoken saved to configuration file: C:\\Users\\SATULURI VINAY\\AppData\\Local/ngrok/ngrok.yml\n"
     ]
    }
   ],
   "source": [
    "!ngrok authtoken 2tIMrVaK80DycXI9ONQa8vr7frN_2BEFZguJtGX8bZEfXBn3H"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fc64cee2-7f46-4100-ba5f-42c2c3e0f6eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🚀 Public URL: https://a380-117-231-192-195.ngrok-free.app\n"
     ]
    }
   ],
   "source": [
    "from pyngrok import ngrok\n",
    "\n",
    "# Set ngrok authentication token\n",
    "ngrok.set_auth_token(\"2tIMrVaK80DycXI9ONQa8vr7frN_2BEFZguJtGX8bZEfXBn3H\")  # Replace with your token\n",
    "\n",
    "# Start ngrok and get the public URL\n",
    "public_url = ngrok.connect(8000).public_url\n",
    "print(f\"🚀 Public URL: {public_url}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "421a0be2-ed6b-4b67-bf69-9546ffdee94c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "INFO:     Started server process [21376]\n",
      "INFO:     Waiting for application startup.\n",
      "INFO:     Application startup complete.\n",
      "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)\n"
     ]
    }
   ],
   "source": [
    "from fastapi import FastAPI\n",
    "import uvicorn\n",
    "import joblib\n",
    "import numpy as np\n",
    "from pydantic import BaseModel\n",
    "import nest_asyncio\n",
    "from pyngrok import ngrok\n",
    "\n",
    "# Initialize FastAPI app\n",
    "app = FastAPI()\n",
    "\n",
    "# Load your trained model\n",
    "model = joblib.load(\"final_fraud_detection_model.pkl\")\n",
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
    "        \"fraud_probability\": round(probability, 4)  # Rounded probability\n",
    "    }\n",
    "\n",
    "# Allow running inside Colab\n",
    "nest_asyncio.apply()\n",
    "\n",
    "# Start FastAPI server\n",
    "uvicorn.run(app, host=\"0.0.0.0\", port=8000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "902fd899-9f55-45f5-9961-034c58f740f9",
   "metadata": {},
   "outputs": [],
   "source": []
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
