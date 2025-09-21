# src/5_fairness.py

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score

print("✔ Loading processed data & models...")

# ───────────────────────────────
# Load artifacts
# ───────────────────────────────
X_test = np.load("processed/X_test.npy")
y_test = np.load("processed/y_test.npy")

models = {
    "catboost": joblib.load("models/catboost_model.pkl"),
    "xgboost": joblib.load("models/xgboost_model.pkl"),
    "tabnet": joblib.load("models/tabnet_model.pkl"),
}

# Load encoders + feature names
label_encoders = joblib.load("processed/label_encoders.pkl")
feature_names = joblib.load("processed/feature_names.pkl")

# Load raw CSV to get sensitive attributes
df = pd.read_csv("data/indian_loans_data.csv")

# Match test indices (IMPORTANT!)
# For simplicity: assume last 20% was test split
test_size = len(y_test)
df_test = df.tail(test_size).reset_index(drop=True)

# Sensitive attributes directly from raw data
sensitive_features = {
    "gender": df_test["gender"].values,
    "marital_status": df_test["marital_status"].values,
    "region": df_test["region"].values,
}

# ───────────────────────────────
# Evaluate Fairness
# ───────────────────────────────
for model_name, model in models.items():
    print(f"\n=== Fairness Metrics: {model_name.upper()} ===")

    # Handle prediction depending on model type
    if model_name == "tabnet":
        y_pred = model.predict(X_test)

    elif model_name == "xgboost":
        dtest = xgb.DMatrix(X_test)
        y_pred_prob = model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)

    else:  # CatBoost
        y_pred = model.predict(X_test)

    # For each sensitive feature
    for feature_name, feature_values in sensitive_features.items():
        mf = MetricFrame(
            metrics={"accuracy": accuracy_score},
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=feature_values
        )

        dp = demographic_parity_difference(y_test, y_pred, sensitive_features=feature_values)
        eo = equalized_odds_difference(y_test, y_pred, sensitive_features=feature_values)

        print(f"Sensitive Feature: {feature_name}")
        print(f"  Accuracy by group: {mf.by_group.to_dict()}")
        print(f"  Demographic Parity Diff: {dp:.4f}")
        print(f"  Equalized Odds Diff: {eo:.4f}")
