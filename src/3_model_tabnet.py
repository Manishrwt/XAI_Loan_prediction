# src/3_model_tabnet.py
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# ---------------------------
# Load preprocessed data
# ---------------------------
X_train = np.load("processed/X_train.npy")
X_test = np.load("processed/X_test.npy")
y_train = np.load("processed/y_train.npy")
y_test = np.load("processed/y_test.npy")

# ---------------------------
# Define and train TabNet model
# ---------------------------
tabnet = TabNetClassifier(
    n_d=32, n_a=32,        # dimensions for decision & attention steps
    n_steps=5,             # number of steps in the architecture
    gamma=1.5,             # relaxation parameter
    lambda_sparse=1e-4,    # sparsity regularization
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=10,
    seed=42
)

tabnet.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_name=["test"],
    eval_metric=["auc"],
    max_epochs=50,
    patience=10,
    batch_size=1024,
    virtual_batch_size=128
)

# ---------------------------
# Evaluate TabNet model
# ---------------------------
y_pred_proba = tabnet.predict_proba(X_test)[:, 1]
y_pred = tabnet.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nTabNet AUC: {auc:.4f}")
print(classification_report(y_test, y_pred, digits=3))

# ---------------------------
# Save model
# ---------------------------
joblib.dump(tabnet, "models/tabnet_model.pkl")
print("✔ Saved models/tabnet_model.pkl")
