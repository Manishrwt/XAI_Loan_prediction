# src/5_fairness.py
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt

# Fairness metrics
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional: AIF360 (skip if not available)
try:
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.datasets import BinaryLabelDataset
    AIF_AVAILABLE = True
except Exception:
    print("⚠️ AIF360 not installed. Skipping AIF metrics.")
    AIF_AVAILABLE = False

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

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

df = pd.read_csv("data/indian_loans_data.csv")

# Assume last 20% = test split
test_size = len(y_test)
df_test = df.tail(test_size).reset_index(drop=True)

sensitive_features = {
    "gender": df_test["gender"].values,
    "marital_status": df_test["marital_status"].values,
    "region": df_test["region"].values,
}

# ───────────────────────────────
# Fairness Computation
# ───────────────────────────────
results = []
os.makedirs("reports", exist_ok=True)

for model_name, model in models.items():
    print(f"\n=== Fairness Metrics: {model_name.upper()} ===")

    # Predict
    if model_name == "tabnet":
        y_pred = model.predict(X_test)
    elif model_name == "xgboost":
        dtest = xgb.DMatrix(X_test)
        y_pred_prob = model.predict(dtest)
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")

    for feat, vals in sensitive_features.items():
        mf = MetricFrame(
            metrics={
                "accuracy": accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "f1": f1_score,
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=vals
        )

        dp = demographic_parity_difference(y_test, y_pred, sensitive_features=vals)
        eo = equalized_odds_difference(y_test, y_pred, sensitive_features=vals)

        print(f"\nSensitive Feature → {feat}")
        print(f"  Accuracy by group: {mf.by_group['accuracy'].to_dict()}")
        print(f"  Demographic Parity Diff: {dp:.4f}")
        print(f"  Equalized Odds Diff: {eo:.4f}")

        # Collect
        results.append({
            "Model": model_name,
            "Sensitive_Feature": feat,
            "Overall_Accuracy": acc,
            "Demographic_Parity_Difference": dp,
            "Equalized_Odds_Difference": eo,
            "Avg_F1": mf.overall["f1"]
        })

# ───────────────────────────────
# Save as CSV
# ───────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df_res = pd.DataFrame(results)
csv_path = f"reports/fairness_metrics_{timestamp}.csv"
df_res.to_csv(csv_path, index=False)
print(f"\n📊 Fairness metrics saved to: {csv_path}")

# ───────────────────────────────
# Generate Fairness Plot (PNG)
# ───────────────────────────────
plt.figure(figsize=(10,6))
for model in df_res["Model"].unique():
    subset = df_res[df_res["Model"] == model]
    plt.bar(
        subset["Sensitive_Feature"] + "_" + model,
        subset["Demographic_Parity_Difference"],
        label=model
    )
plt.axhline(0.1, color='r', linestyle='--', label='Bias threshold (0.1)')
plt.axhline(-0.1, color='r', linestyle='--')
plt.title("Demographic Parity Difference by Model & Feature")
plt.ylabel("DP Difference (lower = fairer)")
plt.xticks(rotation=30, ha='right')
plt.legend()
img_path = f"reports/fairness_dashboard_{timestamp}.png"
plt.tight_layout()
plt.savefig(img_path, dpi=300)
plt.close()
print(f"🖼️ Fairness dashboard image saved: {img_path}")

# ───────────────────────────────
# PDF Summary Report
# ───────────────────────────────
pdf_path = f"reports/fairness_report_{timestamp}.pdf"
c = canvas.Canvas(pdf_path, pagesize=A4)
W, H = A4
x, y = 2*cm, H - 2.5*cm

c.setFont("Helvetica-Bold", 18)
c.setFillColor(colors.darkblue)
c.drawString(x, y, "Fairness Evaluation Report")
y -= 1.2*cm
c.setFont("Helvetica", 11)
c.setFillColor(colors.black)
c.drawString(x, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
y -= 1.0*cm

for _, row in df_res.iterrows():
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, f"Model: {row['Model'].upper()} | Feature: {row['Sensitive_Feature']}")
    y -= 0.5*cm
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"  Accuracy: {row['Overall_Accuracy']:.3f} | F1: {row['Avg_F1']:.3f}")
    y -= 0.4*cm
    c.drawString(x, y, f"  DP Diff: {row['Demographic_Parity_Difference']:.4f} | EO Diff: {row['Equalized_Odds_Difference']:.4f}")
    y -= 0.7*cm
    if y < 3*cm:
        c.showPage()
        y = H - 3*cm

c.setFont("Helvetica-Bold", 11)
c.setFillColor(colors.red)
c.drawString(x, y, "Note: |Fairness difference| > 0.1 may indicate bias.")
c.save()

print(f"📄 PDF report generated: {pdf_path}")
print("✅ Fairness evaluation complete.")
