"""
Script 1: Generate synthetic loan data for credit risk modeling.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 10000

data = {
    "loan_amount": np.random.randint(1000, 50000, n_samples),
    "annual_income": np.random.randint(20000, 150000, n_samples),
    "credit_score": np.random.randint(300, 850, n_samples),
    "employment_years": np.random.randint(0, 30, n_samples),
    "debt_to_income": np.round(np.random.uniform(0.05, 0.75, n_samples), 2),
    "num_credit_lines": np.random.randint(1, 20, n_samples),
    "num_delinquencies": np.random.randint(0, 10, n_samples),
    "loan_purpose": np.random.choice(
        ["home", "auto", "education", "personal", "business"], n_samples
    ),
}

df = pd.DataFrame(data)

# Derive default label based on risk factors
risk_score = (
    (df["credit_score"] < 600).astype(int) * 2
    + (df["debt_to_income"] > 0.5).astype(int) * 2
    + (df["num_delinquencies"] > 3).astype(int) * 2
    + (df["employment_years"] < 2).astype(int)
    + (df["annual_income"] < 30000).astype(int)
)
default_prob = risk_score / risk_score.max()
df["default"] = (np.random.uniform(0, 1, n_samples) < default_prob).astype(int)

df.to_csv("loan_data.csv", index=False)
print(f"Dataset generated: {len(df)} rows, default rate: {df['default'].mean():.2%}")
