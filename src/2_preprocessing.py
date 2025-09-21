"""
Preprocess:
- split train/test
- auto-detect categorical vs numeric
- label-encode categoricals
- scale numerics
- save artifacts: scaler.pkl, label_encoders.pkl, feature_names.pkl
- save train/test matrices for fast model training

Run:
    python src/2_preprocessing.py
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = os.path.join("data", "indian_loans_data.csv")
ART_DIR = "processed"
SCALER_P = os.path.join(ART_DIR, "scaler.pkl")
ENC_P = os.path.join(ART_DIR, "label_encoders.pkl")
FEATS_P = os.path.join(ART_DIR, "feature_names.pkl")
META_P = os.path.join(ART_DIR, "meta.json")

# Keep in sync with 1_data_loader
CANDIDATE_TARGETS = ["Loan_Status", "loan_status", "Approved", "approved", "Target", "target"]
POSITIVE_LABELS = {"Y","Yes","Approved",1,True,"1","True","T"}

def detect_target(df: pd.DataFrame) -> str:
    for cand in CANDIDATE_TARGETS:
        if cand in df.columns:
            return cand
    return df.columns[-1]

def main():
    os.makedirs(ART_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    target_col = detect_target(df)
    y_raw = df[target_col]
    X = df.drop(columns=[target_col]).copy()

    # Detect types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Encoders
    label_encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = X[c].astype(str).fillna("NA")
        X[c] = le.fit_transform(X[c])
        label_encoders[c] = le

    # Target to {0,1}
    if y_raw.dtype == "object":
        y = y_raw.fillna("NA").apply(lambda v: 1 if str(v) in POSITIVE_LABELS else 0).astype(int)
    else:
        y = y_raw.apply(lambda v: 1 if v in POSITIVE_LABELS or v==1 else 0).astype(int)

    # Fill numeric NaNs
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

    # Scale numerics
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
    )

    # Save artifacts
    joblib.dump(scaler, SCALER_P)
    joblib.dump(label_encoders, ENC_P)
    joblib.dump(list(X.columns), FEATS_P)

    # Save matrices for training speed
    np.save(os.path.join(ART_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(ART_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(ART_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(ART_DIR, "y_test.npy"), y_test)

    print("✔ Preprocessing complete")
    print("  Artifacts:", [SCALER_P, ENC_P, FEATS_P])
    print("  Matrices: processed/X_train.npy, X_test.npy, y_train.npy, y_test.npy")
    print("👉 Next: run any 3_model_*.py (CatBoost/LightGBM/XGBoost/TabNet)")

if __name__ == "__main__":
    main()
