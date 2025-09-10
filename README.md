# Credit Card Fraud Detection

## What is this?
A compact, end-to-end **credit-card fraud detection** project. It takes raw transactions, explores the data, builds and tunes a classifier to spot fraud, and serves predictions via a simple REST API. The focus is on **recall-first** (catching more fraud) with a configurable decision threshold.

## Why I built it
To demonstrate the full **data science workflow** on a real, high-imbalance problem that matters to finance teams: from EDA and feature engineering to model selection, evaluation, and deployment. The goal is to show practical trade-offs (precision vs. recall) and how they translate to business impact.

## How it works
- **EDA:** class imbalance, amount/time patterns, fraud by hour/amount bins  
- **Features:** `Hour_of_day`, `Day`, `log(Amount)` (+ original `V1..V28`)  
- **Models:** Logistic Regression, Decision Tree, XGBoost → tuned for **recall**  
- **Final choice:** Decision Tree (test ROC-AUC ≈ 0.90, recall ≈ 0.86 at chosen threshold)  
- **Serving:** saved artifacts and a **Flask API** that computes features and applies a configurable threshold


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
