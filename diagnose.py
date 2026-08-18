import json
import joblib
import pandas as pd
from features import FEATURE_ORDER, CATEGORICAL_FEATURES, NUMERIC_FEATURES, LABEL

df = pd.read_csv("loan_data.csv")

# --- 1. How many defective rows? ---
print("=== OUTLIER SCAN (values > 100x the 75th percentile) ===")
for col in ["principal", "emi_amount", "prior_principal_sum",
            "loan_to_income", "emi_to_income_monthly"]:
    cutoff = df[col].quantile(0.75) * 100
    bad = df[df[col] > cutoff]
    print(f"{col:24s} cutoff {cutoff:>18,.0f}   rows over: {len(bad):>4}")
print()

# --- 2. Score every loan, pick real applicants ---
model = joblib.load("credit_risk_model.pkl")
encoders = joblib.load("label_encoders.pkl")

X = df[FEATURE_ORDER].copy()
for col in CATEGORICAL_FEATURES:
    X[col] = encoders[col].transform(X[col].astype(str))
X = X.astype("float64")

df["prob"] = model.predict_proba(X)[:, 1]

picks = {
    "LOW_RISK":      df[df.is_ntc == 0].nsmallest(1, "prob").iloc[0],
    "HIGH_RISK":     df[df.is_ntc == 0].nlargest(1, "prob").iloc[0],
    "NEW_TO_CREDIT": df[df.is_ntc == 1].nlargest(1, "prob").iloc[0],
}

print("=== REAL APPLICANTS FROM THE GOLD TABLE ===")
for name, row in picks.items():
    print(f"\n{name}  prob={row['prob']:.4f}  actual_default={int(row[LABEL])}")
    payload = {f: (None if pd.isna(row[f]) else
                   (row[f] if f in CATEGORICAL_FEATURES else float(row[f])))
               for f in FEATURE_ORDER}
    print(json.dumps(payload, indent=2, default=str))
