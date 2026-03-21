"""
Script 3: Serve the credit risk model as a Flask REST API.
"""

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

try:
    model = joblib.load("credit_risk_model.pkl")
    le = joblib.load("label_encoder.pkl")
except FileNotFoundError as e:
    raise SystemExit(
        f"ERROR: Model artifact not found ({e}). "
        "Run 01_generate_data.py then 02_train_model.py first."
    ) from e

# Disable XGBoost's internal thread pool.
# Gunicorn forks workers after the model is loaded; XGBoost's OpenMP thread
# pool does not survive a fork and can deadlock on the first predict call.
# For single-row inference there is no performance cost to nthread=1.
model.set_params(nthread=1)

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
    if not payload:
        return jsonify({"error": "Request body must be JSON"}), 400

    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        loan_purpose_encoded = le.transform([payload["loan_purpose"]])[0]
    except ValueError:
        return jsonify({"error": f"Invalid loan_purpose. Valid values: {list(le.classes_)}"}), 400

    try:
        values = [
            payload[f] if f != "loan_purpose" else loan_purpose_encoded
            for f in FEATURE_ORDER
        ]
        prob = model.predict_proba(np.array([values], dtype=np.float64))[0][1]
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
    import socket

    # --- Diagnostics (visible in Application logs) ---
    print("=== CML PORT DIAGNOSTICS ===")
    for var in ("CDSW_APP_PORT", "CDSW_READONLY_PORT", "CDSW_ENGINE_TYPE", "CDSW_PUBLIC_PORT"):
        print(f"  {var} = {os.environ.get(var, 'NOT SET')}")

    def _port_free(port):
        """Return True if we can bind to the port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    # Use CDSW_APP_PORT unless it's already taken by CML infrastructure,
    # in which case fall back to the next available port.
    readonly_port = int(os.environ.get("CDSW_READONLY_PORT", 0))
    requested_port = int(os.environ.get("CDSW_APP_PORT", 5000))

    if requested_port == readonly_port or not _port_free(requested_port):
        print(f"  WARNING: port {requested_port} in use or conflicts with CDSW_READONLY_PORT")
        # Try common fallback ports
        for candidate in (5000, 5001, 9090, 9091):
            if _port_free(candidate):
                requested_port = candidate
                break

    port = requested_port
    print(f"  Binding on port: {port}")
    print("============================")

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
