"""
Script 5: Validate the trained model meets KPI thresholds.

Used as CML Job 3 in the CI/CD pipeline.

Exit codes:
  0 — all thresholds met (pipeline green)
  1 — one or more thresholds missed (pipeline red)
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── KPI thresholds ────────────────────────────────────────────────────────────
# Calibrated to realistic performance on synthetic data.
# Raise these thresholds when switching to real loan data.
ROC_AUC_MIN    = 0.72
F1_DEFAULT_MIN = 0.50   # F1 for class 1 (default) — the business-critical class

# ── Load data (same preprocessing as 02_train_model.py) ──────────────────────
try:
    df = pd.read_csv("loan_data.csv")
except FileNotFoundError:
    print("ERROR: loan_data.csv not found. Run 01_generate_data.py first.")
    sys.exit(1)

le = LabelEncoder()
df["loan_purpose"] = le.fit_transform(df["loan_purpose"])

X = df.drop(columns=["default"])
y = df["default"]

# Identical split to training so we evaluate on the held-out test set
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model = joblib.load("credit_risk_model.pkl")
    model.set_params(nthread=1)
except FileNotFoundError:
    print("ERROR: credit_risk_model.pkl not found. Run 02_train_model.py first.")
    sys.exit(1)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

roc_auc   = roc_auc_score(y_test, y_prob)
f1_default = f1_score(y_test, y_pred)     # F1 for class 1 by default

print(classification_report(y_test, y_pred))
print(f"ROC-AUC        : {roc_auc:.4f}   (min: {ROC_AUC_MIN})")
print(f"F1 (default)   : {f1_default:.4f}   (min: {F1_DEFAULT_MIN})")
print()

# ── Gate ──────────────────────────────────────────────────────────────────────
failures = []
if roc_auc < ROC_AUC_MIN:
    failures.append(f"ROC-AUC {roc_auc:.4f} < {ROC_AUC_MIN}")
if f1_default < F1_DEFAULT_MIN:
    failures.append(f"F1 {f1_default:.4f} < {F1_DEFAULT_MIN}")

if failures:
    print(f"VALIDATION FAILED: {', '.join(failures)}")
    sys.exit(1)

print("VALIDATION PASSED — model meets all KPI thresholds.")
