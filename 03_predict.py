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
    import signal
    import time

    port = int(os.environ.get("CDSW_APP_PORT", 5000))

    def _free_port(port):
        """Kill any process holding the given port, skipping ourselves and our parent."""
        hex_port = format(port, '04X')
        my_pid = os.getpid()
        my_ppid = os.getppid()
        safe_pids = {my_pid, my_ppid}
        killed = False
        for tcp_file in ('/proc/net/tcp', '/proc/net/tcp6'):
            try:
                with open(tcp_file) as f:
                    for line in f.readlines()[1:]:
                        parts = line.strip().split()
                        if len(parts) > 9 and parts[1].split(':')[1].upper() == hex_port:
                            inode = int(parts[9])
                            for pid in os.listdir('/proc'):
                                if not pid.isdigit():
                                    continue
                                pid_int = int(pid)
                                if pid_int in safe_pids:
                                    continue
                                try:
                                    for fd in os.listdir(f'/proc/{pid}/fd'):
                                        try:
                                            if f'socket:[{inode}]' in os.readlink(f'/proc/{pid}/fd/{fd}'):
                                                os.kill(pid_int, signal.SIGKILL)
                                                killed = True
                                        except OSError:
                                            pass
                                except OSError:
                                    pass
            except Exception:
                pass
        if killed:
            time.sleep(2)  # Allow OS to fully release the port

    _free_port(port)

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
