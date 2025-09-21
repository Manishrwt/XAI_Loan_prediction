# src/3_model_xgboost.py
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score

# ---------------------------
# Load preprocessed data
# ---------------------------
X_train = np.load("processed/X_train.npy")
X_test = np.load("processed/X_test.npy")
y_train = np.load("processed/y_train.npy")
y_test = np.load("processed/y_test.npy")

# ---------------------------
# Train XGBoost model
# ---------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 1,       # prevent libomp issues on macOS
    "seed": 42
}

evals = [(dtrain, "train"), (dtest, "test")]
xgb_model = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=20)

# ---------------------------
# Evaluate
# ---------------------------
y_pred_proba = xgb_model.predict(dtest)
y_pred = (y_pred_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nXGBoost AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, digits=3))

# ---------------------------
# Save model
# ---------------------------
joblib.dump(xgb_model, "models/xgboost_model.pkl")
print("✔ Saved models/xgboost_model.pkl")
