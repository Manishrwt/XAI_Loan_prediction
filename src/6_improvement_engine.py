# src/6_improvement_engine.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

# Paths
PROCESSED_DIR = "processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load processed data
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

# Models to evaluate
models = {}
for name in ["catboost_model.pkl", "tabnet_model.pkl", "xgboost_model.pkl"]:
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        models[name.split("_")[0]] = joblib.load(path)

results = []

print("✔ Evaluating models...\n")

for model_name, model in models.items():
    # Handle XGBoost separately (needs DMatrix if saved with Booster API)
    if model_name == "xgboost":
        dtest = xgb.DMatrix(X_test)
        y_pred_prob = model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)
    elif model_name == "tabnet":
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:  # CatBoost and sklearn-compatible
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC": auc
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

print("=== Model Comparison ===")
print(df_results)
print("\n✔ Saved reports/model_comparison.csv")

# Print best model by AUC
best_model = df_results.loc[df_results["AUC"].idxmax()]
print(f"\n🏆 Best Model: {best_model['Model']} (AUC={best_model['AUC']:.4f})")
