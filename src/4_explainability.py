# src/4_explainability.py
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os

# ---------------------------
# Load data and artifacts
# ---------------------------
X_train = np.load("processed/X_train.npy")
X_test = np.load("processed/X_test.npy")
y_train = np.load("processed/y_train.npy")
y_test = np.load("processed/y_test.npy")

scaler = joblib.load("processed/scaler.pkl")
feature_names = joblib.load("processed/feature_names.pkl")

# Load models
catboost_model = joblib.load("models/catboost_model.pkl")
tabnet_model = joblib.load("models/tabnet_model.pkl")
xgboost_model = joblib.load("models/xgboost_model.pkl")

print("✔ Models & data loaded for explainability")

# ---------------------------
# SHAP Explainability
# ---------------------------
print("\n=== SHAP Explanations (CatBoost) ===")
explainer_cb = shap.TreeExplainer(catboost_model)
shap_values_cb = explainer_cb.shap_values(X_test[:100])  # limit for speed

# Summary plot (global feature importance)
plt.title("CatBoost - SHAP Summary")
shap.summary_plot(shap_values_cb, X_test[:100], feature_names=feature_names, show=False)
plt.savefig("reports/shap_catboost_summary.png", bbox_inches="tight")
plt.close()

print("✔ SHAP summary saved: reports/shap_catboost_summary.png")

# ---------------------------
# LIME Explainability
# ---------------------------
print("\n=== LIME Explanations (CatBoost) ===")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=["Rejected", "Approved"],
    mode="classification"
)

# Pick a random test sample
idx = 5
exp = lime_explainer.explain_instance(
    X_test[idx],
    catboost_model.predict_proba
)

exp.save_to_file("reports/lime_example_catboost.html")
print("✔ LIME explanation saved: reports/lime_example_catboost.html")

# ---------------------------
# SHAP for XGBoost
# ---------------------------
print("\n=== SHAP Explanations (XGBoost) ===")
import xgboost as xgb
dtest = xgb.DMatrix(X_test, label=y_test)
explainer_xgb = shap.TreeExplainer(xgboost_model)
shap_values_xgb = explainer_xgb.shap_values(X_test[:100])

plt.title("XGBoost - SHAP Summary")
shap.summary_plot(shap_values_xgb, X_test[:100], feature_names=feature_names, show=False)
plt.savefig("reports/shap_xgboost_summary.png", bbox_inches="tight")
plt.close()

print("✔ SHAP summary saved: reports/shap_xgboost_summary.png")

# ---------------------------
# SHAP for TabNet
# ---------------------------
print("\n=== SHAP Explanations (TabNet) ===")
from pytorch_tabnet.tab_model import TabNetClassifier

# TabNet has its own feature importances
tabnet_importances = tabnet_model.feature_importances_
shap_values_tb = tabnet_model.explain(X_test[:100])

# Bar plot of TabNet feature importances
plt.barh(feature_names, tabnet_importances)
plt.title("TabNet - Feature Importances")
plt.savefig("reports/tabnet_feature_importances.png", bbox_inches="tight")
plt.close()

print("✔ TabNet feature importance saved: reports/tabnet_feature_importances.png")

print("\n✅ Explainability completed. Check the 'reports/' folder for SHAP & LIME outputs.")
