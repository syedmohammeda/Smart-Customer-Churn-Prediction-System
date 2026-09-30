"""Run one example prediction from the command line."""

from __future__ import annotations

import joblib
import pandas as pd

from app import MODEL_PATH


SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 350.0,
}


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run python train_model.py first.")
    model = joblib.load(MODEL_PATH)
    customer = pd.DataFrame([SAMPLE_CUSTOMER], columns=model.feature_names_in_)
    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0, 1])
    print("Customer Churn Prediction")
    print("-------------------------")
    print(f"Prediction: {'Yes' if prediction else 'No'}")
    print(f"Churn Probability: {probability:.2%}")


if __name__ == "__main__":
    main()