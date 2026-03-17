"""
Script 4: Test the credit risk prediction API with sample requests.
"""

import requests

BASE_URL = "http://localhost:5000"

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

print("Health check:", requests.get(f"{BASE_URL}/health").json())
print()

for case in test_cases:
    resp = requests.post(f"{BASE_URL}/predict", json=case["payload"])
    result = resp.json()
    print(f"[{case['description']}]")
    print(f"  Default probability : {result.get('default_probability')}")
    print(f"  Prediction          : {result.get('prediction')}")
    print(f"  Risk label          : {result.get('risk_label')}")
    print()
