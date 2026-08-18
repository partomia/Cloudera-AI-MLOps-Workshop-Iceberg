"""
Script 5: Validate the trained model against credit-risk KPI gates.

Used as CML Job 3 in the CI/CD pipeline.

Gates are Gini and KS, not ROC-AUC and F1. At a 4.22% base rate an F1 gate on
the default class fails a perfectly good model — the baseline run scored 0.96
accuracy while catching zero defaults. Gini and KS measure ranking, which is
what a credit decisioning system actually uses.

Thresholds sit below the observed bootstrap range, not at the median, so
ordinary seed variation does not turn the pipeline red.

Exit codes:
  0 — all gates met (pipeline green)
  1 — one or more gates missed (pipeline red)
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from features import FEATURE_ORDER, CATEGORICAL_FEATURES, LABEL

# ── KPI gates ─────────────────────────────────────────────────────────────────
GINI_MIN = 0.35   # industry floor for a usable retail scorecard
KS_MIN   = 25.0   # separation between good and bad distributions

# Segment gate: the portfolio number can pass while a quarter of the book fails.
# New-to-credit borrowers are 25% of this portfolio and default at 2.4x the rate.
SEGMENT_GINI_MIN = 0.30
SEGMENT_GATE_BLOCKING = False   # report only for now; see note at the end

try:
    df = pd.read_csv("loan_data.csv")
except FileNotFoundError:
    print("ERROR: loan_data.csv not found. Run 01_load_gold.py first.")
    sys.exit(1)

try:
    model = joblib.load("credit_risk_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    model.set_params(nthread=1)
except FileNotFoundError as exc:
    print(f"ERROR: {exc}. Run 02_train_model.py first.")
    sys.exit(1)

for col in CATEGORICAL_FEATURES:
    df[col] = encoders[col].transform(df[col].astype(str))

X = df[FEATURE_ORDER].astype("float64")
y = df[LABEL]

# Identical split to training — same seed, same stratification.
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_prob = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
gini = 2 * roc_auc - 1
fpr, tpr, _ = roc_curve(y_test, y_prob)
ks = float(np.max(tpr - fpr)) * 100

print(f"Holdout        : {len(y_test):,} loans, {int(y_test.sum())} defaults "
      f"({y_test.mean()*100:.2f}%)")
print(f"ROC-AUC        : {roc_auc:.4f}")
print(f"Gini           : {gini:.4f}   (gate: {GINI_MIN})")
print(f"KS             : {ks:.1f}     (gate: {KS_MIN})")
print()

# ── Segment check ─────────────────────────────────────────────────────────────
ntc = X_test["is_ntc"] == 1
segment_failures = []
print("By segment:")
for label, mask in [("New-to-credit", ntc), ("Bureau-backed", ~ntc)]:
    seg_gini = 2 * roc_auc_score(y_test[mask], y_prob[mask]) - 1
    flag = "" if seg_gini >= SEGMENT_GINI_MIN else "  << below segment floor"
    print(f"  {label:14s} n={int(mask.sum()):5,}  "
          f"default={y_test[mask].mean()*100:5.2f}%  Gini={seg_gini:.4f}{flag}")
    if seg_gini < SEGMENT_GINI_MIN:
        segment_failures.append(f"{label} Gini {seg_gini:.4f} < {SEGMENT_GINI_MIN}")
print()

# ── Gate ──────────────────────────────────────────────────────────────────────
failures = []
if gini < GINI_MIN:
    failures.append(f"Gini {gini:.4f} < {GINI_MIN}")
if ks < KS_MIN:
    failures.append(f"KS {ks:.1f} < {KS_MIN}")
if SEGMENT_GATE_BLOCKING:
    failures.extend(segment_failures)

if failures:
    print(f"VALIDATION FAILED: {'; '.join(failures)}")
    sys.exit(1)

if segment_failures:
    print("VALIDATION PASSED (with warnings) — model meets portfolio gates.")
    print(f"  WARNING: {'; '.join(segment_failures)}")
    print("  A portfolio-level pass can mask a segment that would fail on its own.")
else:
    print("VALIDATION PASSED — model meets all KPI gates.")