"""
CML Model Deployment entry point.

Deploy via:  Model Deployments -> New Model
  File     : cml_model.py
  Function : predict

Artifacts must exist in the project filesystem before deployment. Produce them
with the Jobs pipeline:
  Job 1  01_load_gold.py       -> loan_data.csv
  Job 2  02_train_model.py     -> credit_risk_model.pkl, label_encoders.pkl
  Job 3  05_validate_model.py  -> KPI gate

CML imports this module once at startup, so joblib.load runs only on boot.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import warnings
import joblib


from features import DECISION_THRESHOLD, FEATURE_ORDER, build_feature_vector

# __file__ is undefined when CML executes this inside a Jupyter kernel
# (it sends source as a string rather than importing a module).
try:
    _PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _PROJECT_DIR = os.environ.get("CDSW_PROJECT_ROOT", "/home/cdsw")

# Set before XGBoost initialises its OpenMP pool — the pool does not survive
# a fork and can deadlock on the first predict call.
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

try:
    model = joblib.load(os.path.join(_PROJECT_DIR, "credit_risk_model.pkl"))
    encoders = joblib.load(os.path.join(_PROJECT_DIR, "label_encoders.pkl"))
    model.set_params(nthread=1)
    print(f"Model loaded from {_PROJECT_DIR} "
          f"({len(FEATURE_ORDER)} features)", flush=True)
except FileNotFoundError as exc:
    # RuntimeError so CML surfaces this in the deployment log rather than
    # dying on an unhandled SystemExit.
    raise RuntimeError(
        f"Model artifact not found: {exc}. Run the Jobs pipeline before "
        f"deploying (looked in: {_PROJECT_DIR})"
    ) from exc

def predict(args):
    """
    CML Model Deployment handler — called for every POST to the endpoint.

    The PBJ runtime passes the request as a JSON string, while the classic
    CDSW runtime passes a parsed dict. Handle both so this file works on
    either. Some versions also hand over the whole body rather than just
    the `request` field, so unwrap that too.
    """
    if isinstance(args, (str, bytes)):
        try:
            args = json.loads(args)
        except (ValueError, TypeError) as exc:
            return {"error": f"Request body is not valid JSON: {exc}"}

    if not isinstance(args, dict):
        return {"error": f"Expected a JSON object, got {type(args).__name__}"}

    # Unwrap if the full envelope arrived instead of just the request field.
    if "request" in args and isinstance(args["request"], dict):
        args = args["request"]

    vector, error = build_feature_vector(args, encoders)
    if error:
        return {"error": error}

    prob = float(model.predict_proba(vector)[0][1])
    prediction = int(prob >= DECISION_THRESHOLD)

    return {
        "default_probability": round(prob, 4),
        "prediction": prediction,
        "risk_label": "HIGH" if prediction else "LOW",
        "threshold": DECISION_THRESHOLD,
    }