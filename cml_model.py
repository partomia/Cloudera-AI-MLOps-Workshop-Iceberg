"""
CML Model Deployment entry point.

Deploy this file via:  Models → New Model
  File     : cml_model.py
  Function : predict

The model artifacts (credit_risk_model.pkl, label_encoder.pkl) must exist
in the project filesystem before deployment.  Run 01_generate_data.py then
02_train_model.py as CML Jobs (or in a session) to produce them.

CML loads this module once at startup, so joblib.load() runs only on boot.
"""

import os
import warnings
import joblib
import numpy as np

# Absolute path to the project directory — works regardless of the working
# directory CML uses when it boots the model container.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set OMP_NUM_THREADS before importing XGBoost so the OpenMP thread pool
# is never initialised in a way that survives a fork incorrectly.
# This is the same deadlock fix as the Gunicorn path, applied at OS level.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Suppress XGBoost FutureWarnings (deprecated pandas APIs in XGBoost 1.7.x)
# that flood deployment logs without indicating real errors.
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

try:
    model = joblib.load(os.path.join(_PROJECT_DIR, "credit_risk_model.pkl"))
    le    = joblib.load(os.path.join(_PROJECT_DIR, "label_encoder.pkl"))
    model.set_params(nthread=1)
    print(f"Model loaded from {_PROJECT_DIR}", flush=True)
except FileNotFoundError as exc:
    # Raise RuntimeError so CML surfaces the message in the deployment logs
    # rather than silently crashing with an unhandled SystemExit.
    raise RuntimeError(
        f"Model artifact not found: {exc}. "
        f"Run 01_generate_data.py then 02_train_model.py before deploying "
        f"(looked in: {_PROJECT_DIR})"
    ) from exc

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


def predict(args):
    """
    CML Model Deployment handler.

    Called by CML for every POST to the model's /predict endpoint.
    `args` is the `request` field from the JSON body CML receives.

    Example input:
        {
            "loan_amount": 10000, "annual_income": 90000,
            "credit_score": 780,  "employment_years": 10,
            "debt_to_income": 0.15, "num_credit_lines": 5,
            "num_delinquencies": 0, "loan_purpose": "home"
        }
    """
    missing = [f for f in FEATURE_ORDER if f not in args]
    if missing:
        return {"error": f"Missing required fields: {missing}"}

    try:
        loan_purpose_encoded = le.transform([args["loan_purpose"]])[0]
    except ValueError:
        return {"error": f"Invalid loan_purpose. Valid values: {list(le.classes_)}"}

    values = [
        args[f] if f != "loan_purpose" else loan_purpose_encoded
        for f in FEATURE_ORDER
    ]
    prob = model.predict_proba(np.array([values], dtype=np.float64))[0][1]
    prediction = int(prob >= 0.5)

    return {
        "default_probability": round(float(prob), 4),
        "prediction": prediction,
        "risk_label": "HIGH" if prediction == 1 else "LOW",
    }
