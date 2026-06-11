# Customer Churn and Retention Engine

## Objective
This data science project predicts customer churn and evaluates customer value using an end-to-end Machine Learning pipeline. Moving beyond simple classification, this project calculates Customer Lifetime Value (CLV) and performs RFM (Recency, Frequency, Monetary) segmentation to provide actionable business intelligence.

## Features & Methodologies
1. **Data Preparation:** Handled missing data and filtered anomalies in a 500k+ row transactional dataset.
2. **RFM Segmentation:** Grouped customers into cohorts (e.g., *Champions*, *At Risk*, *Lost*) based on purchasing habits.
3. **CLV Modeling (Lifetimes):** Used statistical models (BG/NBD and Gamma-Gamma) to predict future purchases and monetary value probabilistically.
4. **Churn Prediction (XGBoost):** Distilled statistical probabilities into a high-performance XGBoost classifier, achieving **93% Accuracy** and **0.98 ROC-AUC** in predicting customer churn.

## Dataset
Uses the **Online Retail II** dataset from the UCI Machine Learning Repository.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run data pipeline sequentially:
   - `python src/download_data.py`
   - `python src/data_prep.py`
   - `python src/rfm_analysis.py`
   - `python src/clv_modeling.py`
   - `python src/churn_prediction.py`
