import pandas as pd
import os
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

def calculate_clv(input_path, output_path):
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    print("Converting transactions to summary data...")
    # 'lifetimes' requires specific columns: customer_id, datetime, and monetary_value
    summary = summary_data_from_transaction_data(
        df, 
        'Customer ID', 
        'InvoiceDate', 
        monetary_value_col='TotalPrice', 
        observation_period_end=df['InvoiceDate'].max()
    )
    
    # Filter out customers with 0 frequency (only 1 purchase) for Gamma-Gamma model
    summary_returning = summary[summary['frequency'] > 0]
    
    print("Fitting BG/NBD Model...")
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    
    print("Predicting Probability of being Alive (1 - Churn)...")
    summary['p_alive'] = bgf.conditional_probability_alive(
        summary['frequency'], summary['recency'], summary['T']
    )
    # Define Churn Probability
    summary['churn_prob'] = 1 - summary['p_alive']
    
    # Predict number of purchases in next 90 days
    summary['predicted_purchases_90d'] = bgf.predict(
        90, summary['frequency'], summary['recency'], summary['T']
    )
    
    print("Fitting Gamma-Gamma Model...")
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(summary_returning['frequency'], summary_returning['monetary_value'])
    
    print("Predicting Customer Lifetime Value (CLV)...")
    # Predict average monetary value for all customers (even those with freq 0, though rough)
    summary['predicted_monetary_value'] = ggf.conditional_expected_average_profit(
        summary['frequency'], summary['monetary_value']
    )
    
    # Calculate CLV (e.g., over next 12 months, discounted by 0.01 per month)
    summary['CLV_12m'] = ggf.customer_lifetime_value(
        bgf,
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=12, # months
        discount_rate=0.01
    )
    
    print("Top 5 customers by CLV:")
    print(summary.sort_values(by='CLV_12m', ascending=False).head())
    
    print(f"Saving CLV data to {output_path}...")
    summary.to_csv(output_path)
    print("CLV Modeling Done!")

if __name__ == "__main__":
    cleaned_file = os.path.join("data", "processed", "online_retail_cleaned.csv")
    clv_file = os.path.join("data", "processed", "clv_data.csv")
    calculate_clv(cleaned_file, clv_file)
