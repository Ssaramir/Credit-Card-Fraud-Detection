# Credit-Card-Fraud-Detection
# 🛡️ Credit Card Fraud Detection

End-to-end DS project: EDA → features → models → tuning → REST API → simple monitoring.

## What I built
- Processed **284,807** transactions (`Time`, `Amount`, `V1..V28`, `Class`)
- Engineered **Hour_of_day**, **Day**, **log(Amount)**
- Trained LR, Decision Tree, XGBoost; tuned for **recall**
- Final: **Decision Tree** — **recall 0.86**, **ROC-AUC 0.899** (test)
- REST API with configurable threshold + `/health`
- Lightweight Streamlit dashboard (KPIs, recent preds, optional drift)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate      # Win: .venv\Scripts\activate
pip install -r requirements.txt

# Notebook (full walkthrough)
jupyter lab notebooks/fraud_detection.ipynb

# API
python api/fraud_api.py         # http://localhost:5000
curl http://localhost:5000/health
# POST /predict with JSON containing Time (seconds), Amount, V1..V28

# Dashboard
streamlit run monitoring/dashboard.py
Repo
bash
Copy code
notebooks/   # fraud_detection.ipynb
api/         # fraud_api.py
monitoring/  # dashboard.py, predictions_log.csv (auto)
models/      # fraud_model.pkl, feature_names.pkl
data/        # creditcard_sample.csv (tiny demo; link to full dataset)
Results (threshold tuned for recall)
Catches ~86% of fraud (82/95); ~13% missed

Trade-off: ~13 false alarms per true fraud (precision ≈ 7%)

Use threshold/rules to tighten alerts if needed

Dataset
Public “Credit Card Fraud Detection” dataset (anonymized). Link: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>