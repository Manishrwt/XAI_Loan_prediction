"""
3_model_catboost.py
-------------------
Trains a CatBoost model on the loan dataset.
Saves trained model in models/catboost_model.pkl
"""

import os
import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

PROCESSED_DIR = "processed"
MODELS_DIR = "models"

def main():
    # Load processed data
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # Initialize CatBoost
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        verbose=50,
        random_seed=42
    )

    # Train
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nCatBoost AUC: {auc}")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "catboost_model.pkl"))
    print(f"✔ Saved {MODELS_DIR}/catboost_model.pkl")

if __name__ == "__main__":
    main()
