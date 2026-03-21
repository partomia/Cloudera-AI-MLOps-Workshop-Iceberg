"""
Script 4: Test the credit risk prediction API with sample requests.
"""

import os
import sys
import requests

port = os.environ.get("API_PORT", "5000")
BASE_URL = f"http://localhost:{port}"

test_cases = [
    {
        "description": "Low-risk applicant",
        "payload": {
            "loan_amount": 10000,
            "annual_income": 90000,
            "credit_score": 780,
            "employment_years": 10,
            "debt_to_income": 0.15,
            "num_credit_lines": 5,
            "num_delinquencies": 0,
            "loan_purpose": "home",
        },
    },
    {
        "description": "High-risk applicant",
        "payload": {
            "loan_amount": 25000,
            "annual_income": 22000,
            "credit_score": 520,
            "employment_years": 1,
            "debt_to_income": 0.65,
            "num_credit_lines": 2,
            "num_delinquencies": 5,
            "loan_purpose": "personal",
        },
    },
]

try:
    resp = requests.get(f"{BASE_URL}/health", timeout=30)
    resp.raise_for_status()
    print("Health check:", resp.json())
except requests.ConnectionError:
    print(f"ERROR: Cannot connect to API at {BASE_URL}. Is 03_predict.py running?")
    print(f"  If the API started on a different port, set: API_PORT=<port>")
    sys.exit(1)
except requests.RequestException as e:
    print(f"ERROR: Health check failed: {e}")
    sys.exit(1)

print()

for case in test_cases:
    try:
        resp = requests.post(f"{BASE_URL}/predict", json=case["payload"], timeout=30)
        resp.raise_for_status()
        result = resp.json()
        print(f"[{case['description']}]")
        print(f"  Default probability : {result.get('default_probability')}")
        print(f"  Prediction          : {result.get('prediction')}")
        print(f"  Risk label          : {result.get('risk_label')}")
    except requests.RequestException as e:
        print(f"[{case['description']}] ERROR: {e}")
    print()
