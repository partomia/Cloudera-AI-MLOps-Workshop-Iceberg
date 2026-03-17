"""
Script 2: Train an XGBoost credit risk model on the generated loan data.
"""

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

df = pd.read_csv("loan_data.csv")

# Encode categorical feature
le = LabelEncoder()
df["loan_purpose"] = le.fit_transform(df["loan_purpose"])

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

joblib.dump(model, "credit_risk_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model saved to credit_risk_model.pkl")
