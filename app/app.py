# app/app.py
# ============================================================
# Explainable AI for Loan Prediction - Realistic, Text-Only UI
# Bank-style UX, no charts, with dynamic checklists, stress tests,
# product comparison, co-applicants, robust encoders, and rich exports.
# ============================================================

import os
import io
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

# =========================================
# App Config
# =========================================
st.set_page_config(
    page_title="Explainable AI for Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------- CSS (clean, readable, no graphs) -------------
st.markdown("""
<style>
/* rounded info blocks */
.block { padding:14px 18px; border-radius:12px; margin:8px 0; }
.block.ok { background:#113d2f22; border:1px solid #1b5e20; }
.block.warn { background:#26323822; border:1px solid #1c313a; }
.block.bad { background:#3d1b1b22; border:1px solid #8b0000; }
.block.neutral { background:#1e2a3822; border:1px solid #2b394b; }
h2.title-box { background:linear-gradient(90deg, #1E2A38, #2b394b); color:#fff; padding:16px 20px; border-radius:12px; }
.small { font-size: 0.92rem; opacity: 0.92; }
.kv { display:flex; gap:10px; flex-wrap:wrap; }
.kv .item { background:#ffffff08; border:1px dashed #667; padding:8px 10px; border-radius:8px; }
label span.hint { color:#777; font-size:0.85em; margin-left:8px; }
hr.soft { border:0; border-top:1px solid #334; opacity:0.6; margin:8px 0 16px; }
code.mono { background:#111; color:#eee; padding:2px 6px; border-radius:6px; }
.stDownloadButton { margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# =========================================
# Constants / Defaults
# =========================================

# Sensible default rates by loan type for EMI estimation & stress tests (purely heuristic)
RATE_BY_LOAN_TYPE = {
    "Home Loan": 8.5,
    "Personal Loan": 13.5,
    "Business Loan": 12.0,
    "Car Loan": 10.0,
    "Education Loan": 9.0,
    "Other": 11.0
}

# Document checklist by employment & loan/product type
DOCS_BASE = [
    "PAN Card",
    "Aadhaar Card / Valid ID",
    "Recent Passport-size Photograph",
    "Cancelled Cheque / Bank Details",
]

DOCS_SALARIED = [
    "Last 3 Months Salary Slips",
    "Last 6 Months Bank Statements",
    "Last 2 Years Form 16 (if available)",
    "Employment/Offer Letter (if new job)",
]

DOCS_SELF_EMP = [
    "Last 2 Years ITR + Computation",
    "Last 12 Months Bank Statements (Current & Savings)",
    "GST Returns (if applicable)",
    "Business Registration / Shop Act / MSME",
    "Balance Sheet & P&L (CA-audited preferred)",
]

DOCS_HOME = [
    "Property Title / Allotment Letter / Sale Agreement",
    "Chain of Title Documents",
    "Latest Property Tax Receipt",
    "Approved Building Plan / NOC (if applicable)",
]

DOCS_CAR = [
    "Proforma Invoice (Vehicle)",
    "RC Transfer Docs (for used car)",
    "Insurance (if renewal/refinance)"
]

DOCS_EDU = [
    "Admission Letter",
    "Fee Structure / Prospectus",
    "Academic Records (10th/12th/Degree)",
]

DOCS_BUSINESS_LOAN = [
    "Business Vintage Proof",
    "Trade License / Udyam / GST",
    "Debtors/Creditors Summary (if available)",
]

# =========================================
# Load Artifacts
# =========================================
@st.cache_resource
def load_assets():
    scaler = joblib.load("processed/scaler.pkl")
    label_encoders = joblib.load("processed/label_encoders.pkl")
    feature_names = joblib.load("processed/feature_names.pkl")
    model = joblib.load("models/catboost_model.pkl")  # primary model
    return scaler, label_encoders, feature_names, model

scaler, label_encoders, feature_names, model = load_assets()
NUMERIC_COLS = [c for c in feature_names if c not in label_encoders]

# =========================================
# Helpers
# =========================================
def header(title: str, icon: str = "📌"):
    st.markdown(f'<h2 class="title-box">{icon} {title}</h2>', unsafe_allow_html=True)

def _fmt_money(x):
    try:
        return f"₹{float(x):,.0f}"
    except Exception:
        return str(x)

def _ratio(n, d, default=0.0):
    try:
        n = float(n); d = float(d)
        return (n / d) if d else default
    except Exception:
        return default

def _bool(x):  # turns numeric or string to bool-ish
    if isinstance(x, (int, float)): return x > 0
    if isinstance(x, str): return x.strip().lower() in {"true","yes","y","1"}
    return bool(x)

def _get_default_rate(loan_type: str) -> float:
    return RATE_BY_LOAN_TYPE.get(loan_type, RATE_BY_LOAN_TYPE["Other"])

def _parse_float(x, fallback=0.0):
    try:
        return float(x)
    except Exception:
        return fallback

# ---------- EMI helpers ----------
def compute_emi(loan_amount: float, annual_rate: float, years: float):
    r = (annual_rate / 100.0) / 12.0
    n = max(int(years * 12), 1)
    if annual_rate == 0:
        emi = loan_amount / n
    else:
        emi = (loan_amount * r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return emi, r, n

def build_amortization_schedule(loan_amount: float, r: float, n: int):
    schedule = []
    if n <= 0:
        return pd.DataFrame(columns=["Month", "EMI", "Principal", "Interest", "Balance"])
    balance = loan_amount
    emi = (loan_amount * r * (1 + r) ** n) / ((1 + r) ** n - 1) if r > 0 else loan_amount / n
    for m in range(1, n + 1):
        interest = balance * r
        principal = emi - interest
        balance = max(balance - principal, 0)
        schedule.append([m, emi, principal, interest, balance])
    df = pd.DataFrame(schedule, columns=["Month", "EMI", "Principal", "Interest", "Balance"])
    return df

# ---------- Reasons & Suggestions ----------
def generate_reasons_and_suggestions(features: dict, pred: int):
    reasons = []
    suggestions = []

    # pull values safely
    cibil = _parse_float(features.get("cibil_score", 700), 700.0)
    income = _parse_float(features.get("income_annum", 0.0), 0.0)
    co_income = _parse_float(features.get("co_income", 0.0), 0.0)
    total_income = max(income + co_income, 0.0)

    loan_amount = _parse_float(features.get("loan_amount", 0.0), 0.0)
    loan_term = _parse_float(features.get("loan_term", 0.0), 0.0)  # months/years -> treat as months if long?
    term_years = loan_term if loan_term <= 50 else (loan_term/12.0)
    term_years = max(term_years, 0.0001)

    res_assets = _parse_float(features.get("residential_assets_value", 0.0), 0.0)
    com_assets = _parse_float(features.get("commercial_assets_value", 0.0), 0.0)
    collateral = _parse_float(features.get("collateral_value", 0.0), 0.0)
    emi_ratio = _parse_float(features.get("emi_ratio", 0.0), 0.0)
    defaults = _parse_float(features.get("previous_defaults", 0.0), 0.0)
    yexp = _parse_float(features.get("years_at_current_job", 0.0), 0.0)
    emp_type = str(features.get("employment_type", "Salaried"))
    self_emp = str(features.get("self_employed", "No"))
    age = _parse_float(features.get("age", 21.0), 21.0)
    depend = _parse_float(features.get("no_of_dependents", 0.0), 0.0)

    # derived
    assets_total = res_assets + com_assets + collateral
    loan_to_income = _ratio(loan_amount, total_income, 0.0)
    collateral_ratio = _ratio(collateral, loan_amount, 0.0)

    # ----- NEGATIVE FACTORS -----
    if cibil < 650:
        reasons.append("CIBIL score is below 650, which indicates high credit risk.")
        suggestions.append("Improve CIBIL to 700+ by paying bills on time and lowering credit utilization.")
    elif 650 <= cibil < 700 and pred == 0:
        reasons.append("CIBIL score is borderline for approval.")
        suggestions.append("Target a CIBIL of 720+ before reapplying.")

    if emi_ratio > 0.45:
        reasons.append("EMI-to-income ratio is high (> 45%), indicating repayment stress.")
        suggestions.append("Reduce loan amount or extend tenure to bring EMI ratio below 35%.")
    elif 0.35 < emi_ratio <= 0.45 and pred == 0:
        reasons.append("EMI ratio is slightly higher than ideal.")
        suggestions.append("Aim for EMI ratio under 35% by revising amount or tenure.")

    if defaults > 0:
        reasons.append("Previous loan default(s) reduce lender confidence.")
        suggestions.append("Resolve outstanding defaults and maintain on-time payments for 6–12 months.")

    if total_income <= 0:
        reasons.append("No verifiable income provided.")
        suggestions.append("Submit income proofs such as ITR, salary slips, or bank statements.")
    elif loan_to_income > 0.5:
        reasons.append("Requested loan amount is large relative to your household income.")
        suggestions.append("Consider a lower amount or add a co-applicant with income.")

    if collateral_ratio < 0.25 and loan_amount > 0:
        reasons.append("Collateral value is low relative to loan amount.")
        suggestions.append("Increase collateral, add guarantor, or apply under a secured product.")

    if yexp < 1 and emp_type.lower() != "student":
        reasons.append("Short employment stability at current job (< 1 year).")
        suggestions.append("Reapply after 6–12 months at the same job or provide additional references.")

    if _bool(self_emp) and total_income < 500000:
        reasons.append("Self-employed with low declared income increases risk.")
        suggestions.append("Provide audited financials and bank statements for the last 12 months.")

    if age < 21:
        reasons.append("Applicant age is below minimum typical threshold (21).")
        suggestions.append("Apply after turning 21 or add an eligible co-applicant.")
    if depend >= 4 and total_income < 600000:
        reasons.append("High number of dependents with modest income reduces affordability.")
        suggestions.append("Add co-applicant income or reduce requested amount.")

    if term_years <= 2 and loan_amount > 0 and emi_ratio >= 0.30:
        reasons.append("Very short tenure leads to high EMI burden.")
        suggestions.append("Increase tenure to 5–10 years to lower monthly EMIs.")

    if assets_total < (0.1 * loan_amount) and loan_amount > 0:
        reasons.append("Low overall assets relative to loan amount.")
        suggestions.append("Show additional assets or securities to strengthen the profile.")

    # ----- POSITIVE FACTORS -----
    positives = []
    if cibil >= 750:
        positives.append("Excellent CIBIL score (750+).")
    elif 700 <= cibil < 750:
        positives.append("Good CIBIL score (700–749).")

    if emi_ratio <= 0.30:
        positives.append("Healthy EMI ratio (≤ 30%).")
    if loan_to_income <= 0.3 and loan_amount > 0:
        positives.append("Loan amount is conservative relative to household income.")
    if collateral_ratio >= 0.5:
        positives.append("Strong collateral coverage (≥ 50% of loan).")
    if yexp >= 3:
        positives.append("Stable employment history (≥ 3 years).")
    if total_income >= 1200000:
        positives.append("High income supports repayment capacity.")
    if assets_total >= loan_amount:
        positives.append("Assets sufficiently cover the loan amount.")

    # Tailor to decision:
    if pred == 1:
        good_msgs = positives[:6] if positives else ["Overall financial profile indicates strong repayment ability."]
        reasons = good_msgs  # reasons (why approved)
        suggestions = [
            "Maintain on-time payments to keep your CIBIL score healthy.",
            "Avoid taking multiple new loans/credit cards in a short period.",
            "Keep credit utilization under 30% of limits.",
            "Build an emergency fund of 3–6 months of EMIs.",
        ][:max(3, min(6, len(good_msgs)))]
    else:
        if not reasons:
            reasons.append("Profile does not meet current lending policy thresholds.")
        if not suggestions:
            suggestions.append("Improve credit health and reapply in 3–6 months with updated documents.")
        reasons = reasons[:10]
        suggestions = suggestions[:10]

    return reasons, suggestions

# ---------- PDF Generators ----------
def _pdf_line(c, x, y, txt, dy=0.7*cm, bold=False, size=11):
    if bold:
        c.setFont("Helvetica-Bold", size)
    else:
        c.setFont("Helvetica", size)
    c.drawString(x, y, txt)
    y -= dy
    return y

def build_pdf(applicant_name: str,
              loan_type: str,
              decision: str,
              confidence: float,
              reasons: list[str],
              suggestions: list[str]) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    W, H = A4
    x, y = 2*cm, H - 2.5*cm

    c.setFillColor(colors.darkblue)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(x, y, "Loan Decision Report")
    y -= 1.2*cm
    c.setFillColor(colors.black)

    y = _pdf_line(c, x, y, f"Applicant: {applicant_name}", bold=True)
    y = _pdf_line(c, x, y, f"Loan Type: {loan_type}")
    y = _pdf_line(c, x, y, f"Decision : {decision}")
    y = _pdf_line(c, x, y, f"Confidence: {confidence:.2%}")
    y -= 0.3*cm

    y = _pdf_line(c, x, y, "Reasons:", bold=True)
    for r in reasons[:10]:
        y = _pdf_line(c, x, y, f"• {r}", dy=0.55*cm)
    y -= 0.2*cm

    y = _pdf_line(c, x, y, "Suggestions:", bold=True)
    for s in suggestions[:10]:
        y = _pdf_line(c, x, y, f"• {s}", dy=0.55*cm)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def build_bulk_pdf_from_log(df_log: pd.DataFrame) -> bytes:
    """
    Multi-page PDF with one page per logged application.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    W, H = A4
    x, y0 = 2*cm, H - 2.5*cm

    for _, row in df_log.iterrows():
        y = y0
        c.setFillColor(colors.darkblue)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(x, y, "Loan Decision Report (Bulk)")
        y -= 1.2*cm
        c.setFillColor(colors.black)

        applicant_name = str(row.get("Applicant", "N/A"))
        loan_type = str(row.get("Loan_Type", "N/A"))
        decision = str(row.get("Prediction", "N/A"))
        confidence = _parse_float(row.get("Confidence", 0.0), 0.0)

        reasons = str(row.get("Reasons", "")).split(" | ") if "Reasons" in row else []
        suggestions = str(row.get("Suggestions", "")).split(" | ") if "Suggestions" in row else []

        y = _pdf_line(c, x, y, f"Applicant: {applicant_name}", bold=True)
        y = _pdf_line(c, x, y, f"Loan Type: {loan_type}")
        y = _pdf_line(c, x, y, f"Decision : {decision}")
        y = _pdf_line(c, x, y, f"Confidence: {confidence:.2%}")
        y -= 0.3*cm

        y = _pdf_line(c, x, y, "Reasons:", bold=True)
        for r in reasons[:10]:
            y = _pdf_line(c, x, y, f"• {r}", dy=0.55*cm)
        y -= 0.2*cm

        y = _pdf_line(c, x, y, "Suggestions:", bold=True)
        for s in suggestions[:10]:
            y = _pdf_line(c, x, y, f"• {s}", dy=0.55*cm)

        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ---------- Encoding & Prediction Utilities ----------
def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust encoding: if a categorical value wasn't seen in training,
    map it to 0 (first class) to avoid errors.
    """
    df_enc = df.copy()
    for col, le in label_encoders.items():
        if col in df_enc.columns:
            classes = list(le.classes_)
            mapping = {cls: i for i, cls in enumerate(classes)}
            df_enc[col] = df_enc[col].map(mapping).fillna(0).astype(int)
        else:
            df_enc[col] = 0
    # scale numeric only
    if NUMERIC_COLS:
        for nc in NUMERIC_COLS:
            if nc not in df_enc.columns:
                df_enc[nc] = 0.0
        df_enc[NUMERIC_COLS] = scaler.transform(df_enc[NUMERIC_COLS])
    # ensure training order and drop extras
    for c in feature_names:
        if c not in df_enc.columns:
            df_enc[c] = 0
    df_enc = df_enc[feature_names]
    return df_enc

def log_prediction(applicant_name: str,
                   loan_type: str,
                   decision: str,
                   confidence: float,
                   reasons: list[str],
                   suggestions: list[str],
                   features: dict):
    os.makedirs("reports", exist_ok=True)
    path = "reports/prediction_log.csv"
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "Applicant": applicant_name if applicant_name else "N/A",
        "Loan_Type": loan_type,
        "Prediction": decision,
        "Confidence": round(confidence, 4),
        "Reasons": " | ".join(reasons[:10]),
        "Suggestions": " | ".join(suggestions[:10]),
        "Features": json.dumps(features),
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)

# =========================================
# Sidebar Navigation (order requested)
# =========================================
with st.sidebar:
    selected = option_menu(
        "EXPLAINABLE AI",
        ["Home", "Application Details", "Prediction", "EMI Calculator", "Report"],
        icons=["house", "clipboard-data", "magic", "calculator", "file-earmark-text"],
        menu_icon="bank",
        default_index=0,
    )

    # Quick Eligibility Estimator (free, lightweight)
    st.markdown("### ⚡ Quick Eligibility Check")
    quick_income = st.number_input("Annual Income (₹)", min_value=0.0, step=10000.0, key="qi")
    quick_cibil = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1, value=700, key="qc")
    if st.button("Check Eligibility", key="btn_check_elig"):
        if quick_cibil < 650:
            st.error("❌ Low CIBIL (<650) – unlikely eligible.")
        else:
            # Heuristic: not a lending policy — just to guide users
            max_loan = quick_income * (quick_cibil / 900) * 0.6
            st.success(f"✅ You may be eligible up to: {_fmt_money(max_loan)}")

# =========================================
# Pages
# =========================================
if selected == "Home":
    header("Explainable AI for Loan Prediction", "🏦")
    st.markdown("""
1. **Application Details** – Enter the applicant's information (all features used for the model).  
2. **Prediction** – Get a decision *with clear reasons and actionable suggestions*.  
3. **EMI Calculator** – Estimate monthly payments and download an amortization schedule.  
4. **Report** – View the log and download per-applicant PDF/CSV summaries, plus bulk exports.
    """)
    with st.expander("📎 Notes & Disclaimers"):
        st.markdown("""
- This app provides **explainable**, educational guidance. Actual underwriting varies by lender.
- The **Quick Eligibility** is a rough estimate. Please proceed to **Application Details** for a proper evaluation.
- We do not store documents; only the entered fields + decision logs are saved locally under `reports/`.
        """)

elif selected == "Application Details":
    header("Applicant Information", "📝")

    colA, colB = st.columns([1,1])
    with colA:
        applicant_name = st.text_input("👤 Applicant Name", help="Enter full name as on ID.")
    with colB:
        loan_type = st.selectbox("💳 Loan Type", ["Home Loan", "Personal Loan", "Business Loan", "Car Loan", "Education Loan", "Other"],
                                 help="Select the loan product you’re applying for.")

    st.divider()
    st.markdown("#### Provide feature values")
    st.caption("Tip: Use realistic values to get a meaningful decision. Fields are validated and unknown categories are handled safely.")

    ui_vals = {}
    for col in feature_names:
        if col in label_encoders:
            opts = list(label_encoders[col].classes_)
            ui_vals[col] = st.selectbox(col, opts, key=f"f_{col}")
        else:
            # sensible defaults/hints
            minv = 0.0
            step = 1.0
            placeholder = None
            if col in {"cibil_score"}:
                minv, step, placeholder = 0.0, 1.0, 700
            if col in {"loan_term"}:
                st.markdown('<span class="small">Hint: If you enter >50, it will be treated as months; else years.</span>', unsafe_allow_html=True)
            ui_vals[col] = st.number_input(col, min_value=minv, step=step, value=0.0, key=f"f_{col}")

    # Co-Applicant block (kept + improved)
    st.subheader("👥 Co-Applicant (Optional)")
    ui_vals["co_income"] = st.number_input("Co-Applicant Annual Income (₹)", min_value=0.0, step=10000.0)
    ui_vals["co_cibil"] = st.number_input("Co-Applicant CIBIL Score", min_value=300, max_value=900, step=1, value=700)

    # Save button
    if st.button("💾 Save Applicant Details"):
        st.session_state["applicant"] = {
            "name": applicant_name.strip(),
            "loan_type": loan_type,
            "features": ui_vals
        }
        st.success("✅ Applicant details saved. Continue to the Prediction tab.")

elif selected == "Prediction":
    header("Loan Approval Prediction", "🔮")

    if "applicant" not in st.session_state:
        st.warning("⚠️ Please fill **Application Details** first.")
    else:
        app = st.session_state["applicant"]
        features = app["features"]

        # Decision
        if st.button("Run Prediction"):
            df = pd.DataFrame([features])
            df_scaled = encode_and_scale(df)

            pred = int(model.predict(df_scaled.values)[0])
            proba = float(model.predict_proba(df_scaled.values)[0][1])
            decision = "APPROVED" if pred == 1 else "REJECTED"

            # reasons & suggestions
            reasons, suggestions = generate_reasons_and_suggestions(features, pred)

            # ---------- Decision Summary (text, no graphs)
            st.markdown("### 📋 Decision Summary")
            # EMI estimate uses default rate by product and term/amount if present
            loan_amt = _parse_float(features.get("loan_amount", 0.0), 0.0)
            loan_term = _parse_float(features.get("loan_term", 0.0), 0.0)
            # Treat <=50 as years, else months
            tenure_years = loan_term if loan_term <= 50 else (loan_term / 12.0)
            tenure_years = max(tenure_years, 0.0001)
            est_rate = _get_default_rate(app["loan_type"])
            est_emi, _, _ = compute_emi(loan_amt, est_rate, tenure_years)

            col1, col2 = st.columns([1,1])
            with col1:
                if pred == 1:
                    st.markdown(f'<div class="block ok"><b>Decision:</b> APPROVED ✅</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="block bad"><b>Decision:</b> REJECTED ❌</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="block warn"><b>Confidence:</b> {proba:.2%}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="block neutral"><b>Estimated EMI (@ ~{est_rate:.1f}%):</b> {_fmt_money(est_emi)} / month</div>', unsafe_allow_html=True)
                st.caption("Note: EMI here is an estimate using default rates by loan type and the provided tenure/amount.")

            # ---------- Reasons & Suggestions
            st.markdown("### 🧐 Reasons")
            if reasons:
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.markdown("- No specific reasons derived.")

            st.markdown("### 💡 Suggestions")
            if suggestions:
                for s in suggestions:
                    st.markdown(f"- {s}")
            else:
                st.markdown("- Maintain good credit behavior and stable income documents.")

            # ---------- Document Checklist (dynamic)
            st.markdown("### 🗂️ Required Documents (Suggested)")
            docs = DOCS_BASE[:]
            emp = str(features.get("employment_type", "Salaried")).strip().lower()
            if "self" in emp or "business" in emp:
                docs += DOCS_SELF_EMP
            else:
                docs += DOCS_SALARIED

            lt = app["loan_type"]
            if lt == "Home Loan":
                docs += DOCS_HOME
            elif lt == "Car Loan":
                docs += DOCS_CAR
            elif lt == "Education Loan":
                docs += DOCS_EDU
            elif lt == "Business Loan":
                docs += DOCS_BUSINESS_LOAN

            for d in docs:
                st.markdown(f"- {d}")

            # ---------- Stress Test (+1% / +2%)
            st.markdown("### 🧪 Repayment Stress Test (Text Only)")
            if loan_amt > 0 and tenure_years > 0:
                emi_base, _, _ = compute_emi(loan_amt, est_rate, tenure_years)
                emi_p1, _, _ = compute_emi(loan_amt, est_rate+1.0, tenure_years)
                emi_p2, _, _ = compute_emi(loan_amt, est_rate+2.0, tenure_years)
                st.markdown(f"""
- Current rate ~{est_rate:.1f}% → EMI: **{_fmt_money(emi_base)}**  
- If rate +1.0% → EMI: **{_fmt_money(emi_p1)}**  
- If rate +2.0% → EMI: **{_fmt_money(emi_p2)}**
                """)
            else:
                st.caption("Provide loan amount & tenure for a meaningful stress test.")

            # ---------- Product Comparison (text table, no graphs)
            st.markdown("### 🧾 Loan Product EMI Comparison (Text)")
            compare_products = ["Home Loan", "Personal Loan", "Business Loan", "Car Loan", "Education Loan"]
            comp_rows = []
            for p in compare_products:
                r = _get_default_rate(p)
                emi_p, _, _ = compute_emi(loan_amt or 10_00_000, r, tenure_years or 20.0)
                comp_rows.append([p, f"{r:.1f}%", _fmt_money(emi_p)])
            df_comp = pd.DataFrame(comp_rows, columns=["Product", "Assumed Rate", "EMI (₹/month)"])
            st.table(df_comp)

            # persist in session for immediate download
            st.session_state["last_result"] = {
                "name": app['name'] or "N/A",
                "loan_type": app['loan_type'],
                "decision": decision,
                "confidence": proba,
                "reasons": reasons,
                "suggestions": suggestions,
                "features": features
            }

            # log to CSV
            log_prediction(app['name'], app['loan_type'], decision, proba, reasons, suggestions, features)

            # downloads (current applicant)
            st.markdown("#### ⬇️ Download this decision")
            pdf_bytes = build_pdf(app['name'] or "N/A", app['loan_type'], decision, proba, reasons, suggestions)
            st.download_button("Download PDF", data=pdf_bytes,
                               file_name=f"{(app['name'] or 'applicant').replace(' ','_')}_loan_report.pdf",
                               mime="application/pdf")
            row_csv = pd.DataFrame([{
                "Applicant": app['name'] or "N/A",
                "Loan_Type": app['loan_type'],
                "Prediction": decision,
                "Confidence": round(proba, 4),
                "Reasons": " | ".join(reasons),
                "Suggestions": " | ".join(suggestions),
            }]).to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=row_csv,
                               file_name=f"{(app['name'] or 'applicant').replace(' ','_')}_loan_report.csv",
                               mime="text/csv")

elif selected == "EMI Calculator":
    header("EMI Calculator", "📊")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        loan_type = st.selectbox("💳 Loan Type", ["Home Loan", "Personal Loan", "Business Loan", "Car Loan", "Education Loan", "Other"])
    with c2:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0.0, step=10000.0)
    with c3:
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, step=0.1, value=_get_default_rate(loan_type))
    with c4:
        term_years = st.number_input("Tenure (years)", min_value=0.5, step=0.5, value=20.0)

    # Button for calculation
    if st.button("💡 Calculate EMI"):
        if loan_amount > 0 and term_years > 0 and interest_rate >= 0:
            emi, r, n = compute_emi(loan_amount, interest_rate, term_years)
            st.success(f"**{loan_type} EMI:** {_fmt_money(emi)} per month for {term_years:.1f} years")

            # Amortization schedule
            df_schedule = build_amortization_schedule(loan_amount, r, n)
            st.markdown("##### First 12 months (preview)")
            st.dataframe(df_schedule.head(12), use_container_width=True)

            # Stress text (again here)
            st.markdown("###### Rate Stress (Text Only)")
            emi_p1, _, _ = compute_emi(loan_amount, interest_rate+1.0, term_years)
            emi_p2, _, _ = compute_emi(loan_amount, interest_rate+2.0, term_years)
            st.markdown(f"""
- Current rate {interest_rate:.1f}% → EMI: **{_fmt_money(emi)}**  
- If rate +1.0% → EMI: **{_fmt_money(emi_p1)}**  
- If rate +2.0% → EMI: **{_fmt_money(emi_p2)}**
            """)

            st.download_button(
                "Download Full Schedule (CSV)",
                df_schedule.to_csv(index=False).encode("utf-8"),
                "repayment_schedule.csv",
                "text/csv"
            )
        else:
            st.warning("⚠️ Please fill all fields correctly before calculating EMI.")

elif selected == "Report":
    header("Reports & Logs", "🗂️")

    path = "reports/prediction_log.csv"
    if not os.path.exists(path):
        st.info("No predictions logged yet.")
    else:
        df_log = pd.read_csv(path, on_bad_lines="skip")

        # Ensure expected columns
        expected_cols = ["timestamp", "Applicant", "Loan_Type", "Prediction", "Confidence", "Reasons", "Suggestions", "Features"]
        for col in expected_cols:
            if col not in df_log.columns:
                df_log[col] = None
        df_log = df_log[expected_cols]

        # Replace missing applicant names with "N/A"
        df_log["Applicant"] = df_log["Applicant"].fillna("N/A")

        # Show compact log
        st.markdown("#### Recent Decisions")
        st.dataframe(df_log[["timestamp", "Applicant", "Loan_Type", "Prediction", "Confidence"]],
                     use_container_width=True)

        st.markdown("### Download a specific applicant report")

        # Dropdown to select row
        if len(df_log) == 0:
            st.info("Log is empty after filtering.")
        else:
            indices = list(range(len(df_log)))[::-1]  # latest first
            idx_label_map = {
                i: f"{df_log.loc[i, 'Applicant']} | {df_log.loc[i, 'Loan_Type']} | {df_log.loc[i, 'Prediction']} | {df_log.loc[i, 'timestamp']}"
                for i in indices
            }
            choice = st.selectbox("Select a row", options=indices, format_func=lambda i: idx_label_map[i])

            if st.button("Prepare Download"):
                row = df_log.loc[choice]

                # Parse reasons/suggestions safely
                reasons = str(row.get("Reasons", "")).split(" | ") if "Reasons" in row else []
                suggestions = str(row.get("Suggestions", "")).split(" | ") if "Suggestions" in row else []

                # Generate PDF
                pdf_bytes = build_pdf(
                    applicant_name=str(row.get("Applicant", "N/A")),
                    loan_type=str(row.get("Loan_Type", "N/A")),
                    decision=str(row.get("Prediction", "N/A")),
                    confidence=float(row.get("Confidence", 0.0)),
                    reasons=[r for r in reasons if r],
                    suggestions=[s for s in suggestions if s],
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "Download Selected as PDF",
                        data=pdf_bytes,
                        file_name=f"{str(row.get('Applicant','applicant')).replace(' ','_')}_loan_report.pdf",
                        mime="application/pdf",
                        key="dl_pdf_sel"
                    )
                with c2:
                    csv_bytes = row.to_frame().T.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Selected as CSV",
                        data=csv_bytes,
                        file_name=f"{str(row.get('Applicant','applicant')).replace(' ','_')}_loan_report.csv",
                        mime="text/csv",
                        key="dl_csv_sel"
                    )

        st.markdown("#### Download full log")
        st.download_button(
            "Download Full CSV Log",
            data=df_log.to_csv(index=False).encode("utf-8"),
            file_name="prediction_log.csv",
            mime="text/csv"
        )

        # Bulk exports
        st.markdown("### 📦 Bulk Exports (All Applicants)")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download ALL as CSV",
                data=df_log.to_csv(index=False).encode("utf-8"),
                file_name="all_applicants_log.csv",
                mime="text/csv",
                key="dl_all_csv"
            )
        with c2:
            bulk_pdf = build_bulk_pdf_from_log(df_log)
            st.download_button(
                "Download ALL as PDF (multi-page)",
                data=bulk_pdf,
                file_name="all_applicants_report.pdf",
                mime="application/pdf",
                key="dl_all_pdf"
            )
