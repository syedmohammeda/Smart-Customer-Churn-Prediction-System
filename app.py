"""Flask web application and JSON API for churn predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "churn_model.pkl"
NUMERIC_FIELDS = {"SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"}
LOGGER = logging.getLogger(__name__)
app = Flask(__name__)

try:
    MODEL = joblib.load(MODEL_PATH)
    FEATURE_COLUMNS = list(MODEL.feature_names_in_)
except Exception as error:
    MODEL = None
    FEATURE_COLUMNS = []
    LOGGER.warning("Could not load churn model from %s: %s", MODEL_PATH, error)


def prepare_customer(payload: dict[str, Any]) -> pd.DataFrame:
    """Validate a request and return one row in the model's training schema."""
    missing_fields = [field for field in FEATURE_COLUMNS if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    customer = {field: payload[field] for field in FEATURE_COLUMNS}
    for field in NUMERIC_FIELDS.intersection(FEATURE_COLUMNS):
        try:
            value = float(customer[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{field}' must be a valid number") from error
        if not np.isfinite(value):
            raise ValueError(f"'{field}' must be a finite number")
        if field == "SeniorCitizen" and value not in (0, 1):
            raise ValueError("'SeniorCitizen' must be 0 or 1")
        customer[field] = value
    return pd.DataFrame([customer], columns=FEATURE_COLUMNS)


def predict_customer(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the serialized pipeline and format its prediction."""
    if MODEL is None:
        raise RuntimeError("The model is not available. Run python train_model.py first.")
    customer = prepare_customer(payload)
    predicted_class = int(MODEL.predict(customer)[0])
    churn_probability = float(MODEL.predict_proba(customer)[0, 1])
    prediction = "Yes" if predicted_class == 1 else "No"
    return {
        "prediction": prediction,
        "churn_probability": round(churn_probability, 4),
        "message": "Customer is likely to churn."
        if predicted_class == 1
        else "Customer is not likely to churn.",
    }


@app.get("/")
def home():
    return render_template(
        "index.html",
        model_ready=MODEL is not None,
        result=None,
        error=None if MODEL is not None else "Model unavailable. Run python train_model.py first.",
    )


@app.post("/")
def predict_from_form():
    if MODEL is None:
        return render_template(
            "index.html",
            model_ready=False,
            result=None,
            error="Model unavailable. Run python train_model.py first.",
        ), 503
    try:
        result = predict_customer(request.form.to_dict())
    except ValueError as error:
        return render_template("index.html", model_ready=True, result=None, error=str(error)), 400
    return render_template("index.html", model_ready=True, result=result, error=None)


@app.post("/predict")
def predict_api():
    if MODEL is None:
        return jsonify({"error": "Model unavailable. Run python train_model.py first."}), 503
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    try:
        result = predict_customer(payload)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        LOGGER.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the supplied customer values."}), 400
    return jsonify(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    app.run(host="127.0.0.1", port=5000, debug=False)