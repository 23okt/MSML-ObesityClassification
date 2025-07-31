import json

import joblib
import requests

le = joblib.load("label_encoder.pkl")

MLFLOW_MODEL_URL = "http://127.0.0.1:8000/predict"

data = {
    "dataframe_split": {
        "columns": ["Age", "Height", "Weight", "BMI", "PhysicalActivityLevel", "Gender_Male"],
        "data": [
            [18, 190, 70, 19, 1, 1],
            [70, 195, 65, 20, 1, 0],
        ]
    }
}

headers = {"Content-Type": "application/json"}

response = requests.post(MLFLOW_MODEL_URL, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    predictions = response.json()["predictions"]
    label_predictions = le.inverse_transform(predictions)
    print("Predicted Labels:", label_predictions.tolist())
else:
    print("Failed to get prediction:", response.text)