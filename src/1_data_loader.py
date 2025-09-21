"""
2_preprocessing.py
------------------
Preprocesses the dataset:
- Label encode categorical features
- Scale numeric features
- Train/test split
- Save artifacts
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = "data/indian_loans_data.csv"
PROCESSED_DIR = "processed"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Separate target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Identify categorical vs numerical
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Encode categorical
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Scale numerical
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Ensure processed folder
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save artifacts
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(PROCESSED_DIR, "label_encoders.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(PROCESSED_DIR, "feature_names.pkl"))

    # Save datasets
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print("✔ Preprocessing complete")
    print(f"  Artifacts: ['{PROCESSED_DIR}/scaler.pkl', '{PROCESSED_DIR}/label_encoders.pkl', '{PROCESSED_DIR}/feature_names.pkl']")
    print(f"  Matrices: {PROCESSED_DIR}/X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    print("👉 Next: run any 3_model_*.py (CatBoost/LightGBM/XGBoost/TabNet)")

if __name__ == "__main__":
    main()
