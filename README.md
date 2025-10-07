# 🏦 Explainable AI for Loan Prediction

A **Streamlit-based Loan Prediction Web App** with **Explainable AI (XAI)** features.  
This project simulates a real banking workflow, providing:

- Loan approval prediction using a trained **CatBoost model**  
- **Reasons for approval/rejection** with actionable improvement suggestions  
- **EMI Calculator** with stress tests and loan product comparisons  
- **Co-applicant support** for realistic eligibility scenarios  
- **Dynamic document checklist** based on employment type & loan type  
- **Bulk export of PDF/CSV reports** for multiple applicants  
- Fully modular ML pipeline (`src/` folder with data loading, preprocessing, fairness, explainability, etc.)  

---

## 🚀 Features

✔️ **Loan Approval Prediction** – Enter applicant details, get decision with explainable reasons.  
✔️ **Improvement Suggestions** – Personalized tips to increase approval chances.  
✔️ **Co-Applicant Support** – Combine incomes/CIBIL scores for realistic scenarios.  
✔️ **EMI Calculator** – Estimate monthly payments, amortization schedule, stress test (+1%/+2% rate).  
✔️ **Document Checklist** – Dynamic list of required docs per loan type & employment type.  
✔️ **Reports Module** – Download per-applicant or bulk **PDF/CSV reports**.  
✔️ **Explainability & Fairness** – Modules in `src/` handle SHAP, fairness metrics, and improvements.  

---

## 📂 Project Structure

LOAN_PREDICTION/
│
├── app/
│ └── app.py # Streamlit main app
│
├── data/
│ └── indian_loans_data.csv # Training dataset
│
├── models/
│ └── catboost_model.pkl # Trained CatBoost model
│
├── processed/
│ ├── scaler.pkl
│ ├── label_encoders.pkl
│ └── feature_names.pkl
│
├── reports/ # Prediction logs (auto-generated, ignored in git)
│
├── src/ # Modular ML pipeline
│ ├── 1_data_loader.py
│ ├── 2_preprocessing.py
│ ├── 3_model_catboost.py
│ ├── 3_model_xgboost.py
│ ├── 3_model_tabnet.py
│ ├── 4_explainability.py
│ ├── 5_fairness.py
│ ├── 6_improvement_engine.py
│ └── 7_report_generator.py
│
├── requirements.txt # Dependencies
├── README.md # Project description
└── .env # Environment variables (if needed)



---

## ⚙️ Installation
----- app link -------------
https://xailoanprediction-dty8wafucnhefklxq5dk4b.streamlit.app/
1. Clone the repo:
   ```bash
   git clone https://github.com/Manishrwt/XAI_Loan_prediction.git
   

2. Create & activate virtual environment:

python3 -m venv loan_env
source loan_env/bin/activate   # Mac/Linux
loan_env\Scripts\activate      # Windows

3. Install dependencies:

pip install -r requirements.txt

4. ▶️ Run the App
streamlit run app/app.py


The app will open at http://localhost:8501/

Fill applicant details → check prediction → download reports
