import pandas as pd
import os
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

def calculate_clv():
    input_path = os.path.join("data", "processed", "online_retail_cleaned.csv")
    output_path = os.path.join("data", "processed", "clv_data.csv")
    
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    print("Converting transactions to Lifetimes summary data...")
    # 'lifetimes' requires specific columns: frequency, recency, T (customer age), and monetary_value
    summary = summary_data_from_transaction_data(
        df, 
        'Customer ID', 
        'InvoiceDate', 
        monetary_value_col='TotalPrice', 
        observation_period_end=df['InvoiceDate'].max()
    )
    
    print("Fitting BG/NBD Model (Frequency/Churn)...")
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    
    # Predict Probability of being Alive (1 - Churn)
    summary['p_alive'] = bgf.conditional_probability_alive(
        summary['frequency'], summary['recency'], summary['T']
    )
    # Define our mathematical Churn Probability
    summary['churn_prob'] = 1 - summary['p_alive']
    
    print("Fitting Gamma-Gamma Model (Monetary Value)...")
    # Gamma-Gamma requires frequency > 0 (customers who returned at least once)
    summary_returning = summary[summary['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(summary_returning['frequency'], summary_returning['monetary_value'])
    
    # Predict expected average profit per transaction
    summary['predicted_monetary_value'] = ggf.conditional_expected_average_profit(
        summary['frequency'], summary['monetary_value']
    )
    
    print("Predicting 12-Month Customer Lifetime Value (CLV)...")
    # Calculate CLV over next 12 months, discounted by 0.01 per month
    summary['CLV_12m'] = ggf.customer_lifetime_value(
        bgf,
        summary['frequency'],
        summary['recency'],
        summary['T'],
        summary['monetary_value'],
        time=12, 
        discount_rate=0.01
    )
    
    print("\nTop 5 customers by 12-Month CLV:")
    print(summary.sort_values(by='CLV_12m', ascending=False)[['frequency', 'churn_prob', 'CLV_12m']].head())
    
    print(f"\nSaving CLV data to {output_path}...")
    summary.to_csv(output_path)
    print("CLV Modeling complete!")

if __name__ == "__main__":
    calculate_clv()
