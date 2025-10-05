# src/model_comparison_report.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

REPORT_PATH = "reports/model_comparison_report.pdf"
MODEL_CSV = "reports/model_comparison.csv"

def generate_model_comparison_report():
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Model Comparison Report</b>", styles["Title"]))
    elements.append(Spacer(1, 20))

    # Intro
    elements.append(Paragraph("This report compares the performance of CatBoost, XGBoost, and TabNet models "
                              "for loan approval prediction. Each model was trained and evaluated using "
                              "identical datasets and pre-processing steps.", styles["Normal"]))
    elements.append(Spacer(1, 12))

    if not os.path.exists(MODEL_CSV):
        elements.append(Paragraph("⚠️ model_comparison.csv not found in reports/ directory.", styles["Normal"]))
    else:
        df = pd.read_csv(MODEL_CSV)

        # Table
        elements.append(Paragraph("<b>1. Performance Metrics</b>", styles["Heading2"]))
        table_data = [df.columns.tolist()] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

        # Plot
        numeric_cols = [c for c in df.columns if c.lower() not in ["model", "name"]]
        df.set_index(df.columns[0], inplace=True)

        # Generate bar plot
        plt.figure(figsize=(6, 4))
        df[numeric_cols].plot(kind='bar', figsize=(7, 4))
        plt.title("Model Performance Comparison")
        plt.xlabel("Model")
        plt.ylabel("Metric Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = "reports/model_comparison_chart.png"
        plt.savefig(plot_path)
        plt.close()

        # Insert into PDF
        elements.append(Paragraph("<b>2. Visual Comparison</b>", styles["Heading2"]))
        elements.append(Image(plot_path, width=400, height=250))
        elements.append(Spacer(1, 12))

        # Auto Conclusion
        best_model = df[numeric_cols].mean(axis=1).idxmax()
        elements.append(Paragraph("<b>3. Observations & Recommendation</b>", styles["Heading2"]))
        elements.append(Paragraph(f"The analysis shows that <b>{best_model}</b> achieved the best overall "
                                  "performance across key metrics, making it the preferred model "
                                  "for deployment in the Explainable AI Loan Prediction System.", styles["Normal"]))

    # Build PDF
    doc = SimpleDocTemplate(REPORT_PATH, pagesize=A4)
    doc.build(elements)
    print(f"✅ Model comparison report generated: {REPORT_PATH}")


if __name__ == "__main__":
    generate_model_comparison_report()
