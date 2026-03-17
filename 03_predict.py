"""
Script 3: Serve the credit risk model as a Flask REST API.
"""

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

model = joblib.load("credit_risk_model.pkl")
le = joblib.load("label_encoder.pkl")

FEATURE_ORDER = [
    "loan_amount",
    "annual_income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "num_credit_lines",
    "num_delinquencies",
    "loan_purpose",
]


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    try:
        loan_purpose_encoded = le.transform([payload["loan_purpose"]])[0]
        features = np.array([[
            payload["loan_amount"],
            payload["annual_income"],
            payload["credit_score"],
            payload["employment_years"],
            payload["debt_to_income"],
            payload["num_credit_lines"],
            payload["num_delinquencies"],
            loan_purpose_encoded,
        ]])
        prob = model.predict_proba(features)[0][1]
        prediction = int(prob >= 0.5)
        return jsonify({
            "default_probability": round(float(prob), 4),
            "prediction": prediction,
            "risk_label": "HIGH" if prediction == 1 else "LOW",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
