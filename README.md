# 🛡️ Credit Card Fraud Detection

End-to-end DS project: EDA → features → models → tuning → REST API.

## What I built
- Processed **284,807** transactions (`Time`, `Amount`, `V1..V28`, `Class`)
- Engineered **Hour_of_day**, **Day**, **log(Amount)**
- Trained Logistic Regression, Decision Tree, XGBoost; tuned for **recall**
- Final: **Decision Tree** — **recall 0.86**, **ROC-AUC 0.899** (test)
- REST API with configurable decision threshold + `/health`

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -r requirements.txt

# Notebook
jupyter lab notebooks/fraud_detection.ipynb

# API
python api/fraud_api.py         # http://localhost:5000
curl http://localhost:5000/health
```
Dataset
Public “Credit Card Fraud Detection” dataset. Link: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
