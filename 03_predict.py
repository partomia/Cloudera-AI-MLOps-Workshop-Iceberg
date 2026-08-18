"""
Script 3: Serve the credit risk model as a Flask REST API from a CML session.

Session path — lives only while the session is open. See cml_model.py for the
managed Model Deployment equivalent.
"""

import joblib
from flask import Flask, jsonify, request

from features import DECISION_THRESHOLD, FEATURE_ORDER, build_feature_vector

app = Flask(__name__)

try:
    model = joblib.load("credit_risk_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
except FileNotFoundError as e:
    raise SystemExit(
        f"ERROR: Model artifact not found ({e}). "
        "Run 01_load_gold.py then 02_train_model.py first."
    ) from e

# Gunicorn forks workers after the model loads; XGBoost's OpenMP pool does not
# survive a fork and can deadlock on first predict. No cost for single-row work.
model.set_params(nthread=1)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON"}), 400

    vector, error = build_feature_vector(payload, encoders)
    if error:
        return jsonify({"error": error}), 400

    prob = float(model.predict_proba(vector)[0][1])
    prediction = int(prob >= DECISION_THRESHOLD)

    return jsonify({
        "default_probability": round(prob, 4),
        "prediction": prediction,
        "risk_label": "HIGH" if prediction else "LOW",
        "threshold": DECISION_THRESHOLD,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features": len(FEATURE_ORDER)})


if __name__ == "__main__":
    import os
    import socket

    print("=== CML PORT DIAGNOSTICS ===")
    for var in ("CDSW_APP_PORT", "CDSW_READONLY_PORT", "CDSW_ENGINE_TYPE"):
        print(f"  {var} = {os.environ.get(var, 'NOT SET')}")

    def _port_free(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    readonly_port = int(os.environ.get("CDSW_READONLY_PORT", 0))
    port = int(os.environ.get("CDSW_APP_PORT", 5000))

    if port == readonly_port or not _port_free(port):
        print(f"  WARNING: port {port} unavailable")
        for candidate in (5000, 5001, 9090, 9091):
            if _port_free(candidate):
                port = candidate
                break

    print(f"  Binding on port: {port}")
    print("============================")

    from gunicorn.app.base import BaseApplication

    class StandaloneApp(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for k, v in self.options.items():
                self.cfg.set(k.lower(), v)

        def load(self):
            return self.application

    StandaloneApp(app, {"bind": f"0.0.0.0:{port}", "workers": 2,
                        "timeout": 120}).run()