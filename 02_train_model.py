"""
Script 2: Train an XGBoost credit risk model on the generated loan data.
         Parameters, metrics, and model artifact are logged with MLflow
         when available (CML sessions).  When MLflow is absent (Docker build
         via cdsw-build.sh) training and pkl serialisation still complete
         normally — only experiment tracking is skipped.

In Cloudera AI (CML) the MLFLOW_TRACKING_URI env var is automatically set
to the workspace-level MLflow tracking server — no extra configuration needed.
When running locally the run is stored in ./mlruns.
"""

import contextlib
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# MLflow is pre-installed in CML sessions via mlflow-cml-plugin but is NOT
# available in the Docker build environment (intentionally excluded from
# requirements.txt to avoid breaking CML's pinned mlflow-skinny version).
try:
    import mlflow
    import mlflow.xgboost
    _mlflow_available = True
except ImportError:
    _mlflow_available = False
    print("mlflow not available — skipping experiment tracking", flush=True)

try:
    df = pd.read_csv("loan_data.csv")
except FileNotFoundError:
    raise SystemExit("ERROR: loan_data.csv not found. Run 01_generate_data.py first.")

# Encode categorical feature
le = LabelEncoder()
df["loan_purpose"] = le.fit_transform(df["loan_purpose"])

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameters — defined once so they can be logged to MLflow
PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": 42,
}

if _mlflow_available:
    mlflow.set_experiment("credit-risk-model")
    run_ctx = mlflow.start_run()
else:
    run_ctx = contextlib.nullcontext()

with run_ctx:
    if _mlflow_available:
        # --- Log hyperparameters ---
        mlflow.log_params(PARAMS)
        mlflow.log_params({
            "test_size": 0.2,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "default_rate": round(float(y.mean()), 4),
        })

    # --- Train ---
    model = XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")

    if _mlflow_available:
        # --- Log KPI / success metrics ---
        mlflow.log_metrics({
            "roc_auc":               round(roc_auc, 4),
            "accuracy":              round(report["accuracy"], 4),
            # Class 1 = default (the KPI class)
            "precision_default":     round(report["1"]["precision"], 4),
            "recall_default":        round(report["1"]["recall"], 4),
            "f1_default":            round(report["1"]["f1-score"], 4),
            # Class 0 = no default
            "precision_no_default":  round(report["0"]["precision"], 4),
            "recall_no_default":     round(report["0"]["recall"], 4),
            "f1_no_default":         round(report["0"]["f1-score"], 4),
        })

        # --- Log model artifact with signature and input example ---
        # Use artifact_path (compatible with mlflow-skinny 2.x used by CML's mlflow-cml-plugin)
        input_example = X_test.iloc[:5]
        signature = mlflow.models.infer_signature(X_test, y_prob)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        # --- Log label encoder as a supplementary artifact ---
        joblib.dump(le, "label_encoder.pkl")
        mlflow.log_artifact("label_encoder.pkl", artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow run logged  — run_id: {run_id}")
        print(f"Experiment         : credit-risk-model")
    else:
        # Save label encoder when MLflow is not available (Docker build path)
        joblib.dump(le, "label_encoder.pkl")

# Save pkl for cml_model.py (Model Deployment) and 03_predict.py (session API)
joblib.dump(model, "credit_risk_model.pkl")
print("Model saved to credit_risk_model.pkl")
