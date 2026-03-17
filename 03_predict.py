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
    import os
    port = int(os.environ.get("CDSW_APP_PORT", 5000))
    from gunicorn.app.base import BaseApplication

    class StandaloneApp(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        "bind": f"0.0.0.0:{port}",
        "workers": 2,
        "timeout": 120,
    }
    StandaloneApp(app, options).run()
