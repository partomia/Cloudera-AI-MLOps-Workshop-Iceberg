"""
Script 2: Train an XGBoost credit risk model on the M4 gold feature table.

Run 3 of 3 — widens 8 features to 21. Balancing and split unchanged from
run 2, so the delta is attributable to the added features alone.
"""

import contextlib
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from features import FEATURE_ORDER, CATEGORICAL_FEATURES, NUMERIC_FEATURES, LABEL

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
    raise SystemExit("ERROR: loan_data.csv not found. Run 01_load_gold.py first.")

# One encoder per categorical, saved together so serving can reproduce them.
encoders = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[FEATURE_ORDER].astype("float64")
y = df[LABEL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X = df[FEATURE_ORDER].astype("float64")

n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
SCALE_POS_WEIGHT = n_neg / n_pos

PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": 42,
    "scale_pos_weight": SCALE_POS_WEIGHT,
}

if _mlflow_available:
    mlflow.set_experiment("credit-risk-model")
    run_ctx = mlflow.start_run(run_name="balanced-21-features")
else:
    run_ctx = contextlib.nullcontext()

with run_ctx:
    if _mlflow_available:
        mlflow.log_params(PARAMS)
        mlflow.log_params({
            "test_size": 0.2,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "default_rate_portfolio": round(float(y.mean()), 4),
            "default_rate_holdout": round(float(y_test.mean()), 4),
            "n_features": X.shape[1],
            "balancing": "scale_pos_weight",
        })

    model = XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    gini = 2 * roc_auc - 1
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ks_idx = int(np.argmax(tpr - fpr))
    ks = (tpr - fpr)[ks_idx] * 100
    ks_threshold = float(thresholds[ks_idx])

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"ROC-AUC      : {roc_auc:.4f}")
    print(f"Gini         : {gini:.4f}   (gate: 0.35)")
    print(f"KS           : {ks:.1f}     (gate: 25)")
    print(f"KS threshold : {ks_threshold:.4f}")
    print()

    # --- Feature importance: did the extra 13 features earn their place? ---
    imp = (pd.Series(model.feature_importances_, index=FEATURE_ORDER)
             .sort_values(ascending=False))
    print("Feature importance (top 12):")
    print((imp.head(12) * 100).round(2).to_string())
    print()
    zero_imp = imp[imp == 0].index.tolist()
    if zero_imp:
        print(f"Zero importance ({len(zero_imp)}): {', '.join(zero_imp)}")
        print()

    # --- Risk quintiles. 149 test defaults is too few for deciles. ---
    q = pd.DataFrame({"prob": y_prob, "actual": y_test.values})
    q["quintile"] = pd.qcut(q["prob"].rank(method="first"), 5,
                            labels=["Q1 safest", "Q2", "Q3", "Q4", "Q5 riskiest"])
    lift = q.groupby("quintile", observed=True).agg(
        loans=("actual", "size"), defaults=("actual", "sum"),
    )
    lift["default_rate_%"] = (lift["defaults"] / lift["loans"] * 100).round(2)
    print("Risk quintiles (holdout):")
    print(lift.to_string())
    print()

    # --- NTC segment: does the model rank thin-file borrowers separately? ---
    ntc_mask = X_test["is_ntc"] == 1
    for label, mask in [("New-to-credit", ntc_mask), ("Bureau-backed", ~ntc_mask)]:
        seg_auc = roc_auc_score(y_test[mask], y_prob[mask])
        print(f"{label:14s}: n={int(mask.sum()):5,}  "
              f"default={y_test[mask].mean()*100:5.2f}%  "
              f"Gini={2*seg_auc-1:.4f}")
    print()

    if _mlflow_available:
        mlflow.log_metrics({
            "roc_auc":           round(roc_auc, 4),
            "gini":              round(gini, 4),
            "ks":                round(float(ks), 2),
            "ks_threshold":      round(ks_threshold, 4),
            "accuracy":          round(report["accuracy"], 4),
            "precision_default": round(report["1"]["precision"], 4),
            "recall_default":    round(report["1"]["recall"], 4),
            "f1_default":        round(report["1"]["f1-score"], 4),
        })
        signature = mlflow.models.infer_signature(X_test, y_prob)
        mlflow.xgboost.log_model(
            model, artifact_path="model",
            input_example=X_test.iloc[:5], signature=signature,
        )
        joblib.dump(encoders, "label_encoders.pkl")
        mlflow.log_artifact("label_encoders.pkl", artifact_path="model")
        print(f"MLflow run logged  — run_id: {mlflow.active_run().info.run_id}")
    else:
        joblib.dump(encoders, "label_encoders.pkl")

joblib.dump(model, "credit_risk_model.pkl")
print("Model saved to credit_risk_model.pkl")