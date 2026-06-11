import pandas as pd
import datetime as dt
import os
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

def calculate_clv():
    input_path = os.path.join("data", "processed", "online_retail_cleaned.csv")
    output_path = os.path.join("data", "processed", "clv_data.csv")
    
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    # TEMPORAL SPLIT: Hide the future 90 days from the CLV model too!
    latest_date = df['InvoiceDate'].max()
    cutoff_date = latest_date - dt.timedelta(days=90)
    df_past = df[df['InvoiceDate'] <= cutoff_date]
    
    print("Converting past transactions to Lifetimes summary data...")
    summary = summary_data_from_transaction_data(
        df_past, 
        'Customer ID', 
        'InvoiceDate', 
        monetary_value_col='TotalPrice', 
        observation_period_end=cutoff_date
    )
    
    print("Fitting Models...")
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    
    summary['p_alive'] = bgf.conditional_probability_alive(summary['frequency'], summary['recency'], summary['T'])
    summary['churn_prob'] = 1 - summary['p_alive']
    
    summary_returning = summary[summary['frequency'] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(summary_returning['frequency'], summary_returning['monetary_value'])
    
    summary['predicted_monetary_value'] = ggf.conditional_expected_average_profit(summary['frequency'], summary['monetary_value'])
    summary['CLV_12m'] = ggf.customer_lifetime_value(bgf, summary['frequency'], summary['recency'], summary['T'], summary['monetary_value'], time=12, discount_rate=0.01)
    
    summary.to_csv(output_path)
    print("CLV Modeling complete!")

if __name__ == "__main__":
    calculate_clv()
