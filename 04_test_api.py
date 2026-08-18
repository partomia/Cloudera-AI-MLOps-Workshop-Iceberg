"""
Script 4: Test the credit risk API with real applicants from the gold table.

These are not hypothetical profiles. Each is an actual row from
federal12_gold.credit_risk_features — the table built in M4 — so the payloads
are guaranteed in-distribution and the true outcome is known. The API returns
a prediction; the ACTUAL line says what really happened to that loan.
"""

import os
import sys
import requests

BASE_URL = f"http://localhost:{os.environ.get('API_PORT', '5000')}"

# Lowest-scoring bureau-backed loan in the portfolio. Did not default.
LOW_RISK = {
    "principal": 139938.0, "tenure_months": 36.0, "interest_rate": 0.2392,
    "emi_amount": 5484.0, "annual_income": 2740000.0, "employment_years": 7.0,
    "age_years": 34.0, "credit_score": 775.0, "num_credit_lines": 5.0,
    "num_delinquencies": 0.0, "debt_to_income": 0.14,
    "credit_history_months": 47.0, "credit_utilisation": 0.47, "is_ntc": 0.0,
    "loan_to_income": 0.0510722627737226,
    "emi_to_income_monthly": 0.0240175182481751,
    "prior_loan_count": 2.0, "prior_default_count": 0.0,
    "prior_principal_sum": 4880000.0,
    "employment_status": "self_employed", "purpose": "business",
}

# Highest-scoring bureau-backed loan. Credit score 325, one prior default,
# EMI at 6.8x monthly income. Defaulted.
HIGH_RISK = {
    "principal": 2280000.0, "tenure_months": 12.0, "interest_rate": 0.146,
    "emi_amount": 205359.0, "annual_income": 360000.0, "employment_years": 1.0,
    "age_years": 47.0, "credit_score": 325.0, "num_credit_lines": 7.0,
    "num_delinquencies": 9.0, "debt_to_income": 0.7,
    "credit_history_months": 225.0, "credit_utilisation": 0.51, "is_ntc": 0.0,
    "loan_to_income": 6.333333333333333, "emi_to_income_monthly": 6.8453,
    "prior_loan_count": 1.0, "prior_default_count": 1.0,
    "prior_principal_sum": 3930000.0,
    "employment_status": "salaried", "purpose": "auto",
}

# Highest-scoring new-to-credit loan. No bureau file at all — every bureau
# field is null. The model scores it 0.94 purely on behavioural features
# built in M4: 4 prior loans, 2 prior defaults. Defaulted.
NEW_TO_CREDIT = {
    "principal": 1910000.0, "tenure_months": 36.0, "interest_rate": 0.1,
    "emi_amount": 61630.0, "annual_income": 970000.0, "employment_years": 2.0,
    "age_years": 26.0,
    "credit_score": None, "num_credit_lines": None, "num_delinquencies": None,
    "debt_to_income": None, "credit_history_months": None,
    "credit_utilisation": None,
    "is_ntc": 1.0,
    "loan_to_income": 1.9690721649484535,
    "emi_to_income_monthly": 0.7624329896907217,
    "prior_loan_count": 4.0, "prior_default_count": 2.0,
    "prior_principal_sum": 11900000.0,
    "employment_status": "self_employed", "purpose": "home",
}

CASES = [
    ("Low risk, bureau-backed",       LOW_RISK,      0),
    ("High risk, prior default",      HIGH_RISK,     1),
    ("New-to-credit, no bureau file", NEW_TO_CREDIT, 1),
]

try:
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    r.raise_for_status()
    print("Health check:", r.json(), "\n")
except requests.ConnectionError:
    print(f"ERROR: cannot reach {BASE_URL}. Is 03_predict.py running?")
    print("  If it bound to another port: API_PORT=<port> python 04_test_api.py")
    sys.exit(1)

for name, payload, actual in CASES:
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        r.raise_for_status()
        out = r.json()
        agree = "correct" if out["prediction"] == actual else "MISSED"
        print(f"[{name}]")
        print(f"  Default probability : {out['default_probability']}")
        print(f"  Decision            : {out['risk_label']} "
              f"(threshold {out['threshold']})")
        print(f"  Actual outcome      : "
              f"{'DEFAULTED' if actual else 'repaid'}  -> {agree}")
    except requests.RequestException as e:
        print(f"[{name}] ERROR: {e}")
    print()