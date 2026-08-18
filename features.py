"""
Single source of truth for the model's feature contract.

Every script imports from here. Previously FEATURE_ORDER was duplicated in
03_predict.py and cml_model.py, which is how a serving endpoint silently
starts scoring garbage.
"""

# Bureau fields are NULL for new-to-credit borrowers (4,449 of 17,636 rows).
# They are left as NaN on purpose — XGBoost splits on missingness natively,
# and "no bureau file" is itself predictive. Do not impute these.
NUMERIC_FEATURES = [
    # Loan terms
    "principal", "tenure_months", "interest_rate", "emi_amount",
    # Borrower
    "annual_income", "employment_years", "age_years",
    # Bureau
    "credit_score", "num_credit_lines", "num_delinquencies",
    "debt_to_income", "credit_history_months", "credit_utilisation", "is_ntc",
    # Derived / behavioural
    "loan_to_income", "emi_to_income_monthly",
    "prior_loan_count", "prior_default_count", "prior_principal_sum",
]

CATEGORICAL_FEATURES = ["employment_status", "purpose"]

FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES   # 21 features

LABEL = "default_flag"

SOURCE_TABLE = "federal12_gold.credit_risk_features"
CONNECTION = "federal-impala-1"

# Excluded on purpose:
#   loan_id, customer_id  — identifiers
#   disbursal_date        — used for auditing, not as a feature
#   _gold_ts              — pipeline audit column
#   state                 — highest-cardinality categorical; 149 test defaults
#                           is too few to fit ~28 levels without memorising

import numpy as np

# Threshold from run 3's KS-optimal point, not 0.5.
# The model was trained with scale_pos_weight, so probabilities are inflated
# relative to the 4.22% portfolio base rate — they rank well but are not
# calibrated. The cut-off is a credit policy decision, not a modelling one.
DECISION_THRESHOLD = 0.3793


def build_feature_vector(payload, encoders):
    """
    Turn a JSON payload into a model-ready 1-row array.

    Returns (vector, error). Exactly one is None.

    Bureau fields may be sent as null for new-to-credit borrowers — that is a
    valid, meaningful input, not a malformed request. The key must be present;
    its value may be None.
    """
    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        return None, f"Missing required fields: {missing}"

    values = []
    for f in FEATURE_ORDER:
        raw = payload[f]

        if f in CATEGORICAL_FEATURES:
            if raw is None:
                return None, f"'{f}' cannot be null"
            try:
                values.append(float(encoders[f].transform([str(raw)])[0]))
            except ValueError:
                return None, (f"Invalid {f}: '{raw}'. "
                              f"Valid values: {list(encoders[f].classes_)}")
        else:
            # None -> NaN. XGBoost splits on missingness natively.
            values.append(np.nan if raw is None else float(raw))

    return np.array([values], dtype=np.float64), None