# Customer Churn and Retention Engine

## Objective
This repository contains a full-stack data science project that predicts customer churn and evaluates customer value. Moving beyond simple classification, this project calculates Customer Lifetime Value (CLV) and performs RFM (Recency, Frequency, Monetary) segmentation to provide actionable business intelligence.

## Features & Methodologies
1. **Data Preparation & Cleaning:** Handling missing data, filtering anomalies, and parsing temporal features.
2. **RFM Segmentation:** Grouping customers into cohorts (e.g., *Champions*, *At Risk*, *Lost*) based on their purchasing habits.
3. **CLV Modeling (Lifetimes):** Using statistical models (BG/NBD and Gamma-Gamma) to predict future purchases and monetary value probabilistically.
4. **Churn Prediction (XGBoost):** Distilling the complex statistical probabilities and behavioral features into a high-performance XGBoost classifier to accurately predict customer churn.

## Dataset
The project uses the **Online Retail II** dataset from the UCI Machine Learning Repository. It contains transactional data from an online retailer spanning from 2010 to 2011.

## Project Structure
```text
├── data/
│   ├── raw/                  # Raw downloaded datasets
│   └── processed/            # Cleaned data, RFM and CLV outputs
├── models/                   # Saved machine learning models (XGBoost)
├── src/
│   ├── download_data.py      # Automated data acquisition
│   ├── data_prep.py          # Data cleaning scripts
│   ├── rfm_analysis.py       # RFM segmentation logic
│   └── clv_modeling.py       # Lifetimes and XGBoost pipeline
│   └── churn_prediction.py   # Churn Prediction Model
├── venv/                     # Python Virtual Environment
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd customer-churn-and-retention-engine-project-1
   ```

2. **Set up the virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the pipeline sequentially:**
   ```bash
   python src/download_data.py
   python src/data_prep.py
   python src/rfm_analysis.py
   python src/clv_modeling.py
   python src/churn_prediction.py
   ```

## Technologies Used
* **Python**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, XGBoost
* **Statistical Modeling**: Lifetimes (BG/NBD, Gamma-Gamma)
