import pandas as pd
import datetime as dt
import os

def calculate_rfm():
    input_path = os.path.join("data", "processed", "online_retail_cleaned.csv")
    output_path = os.path.join("data", "processed", "rfm_data.csv")
    
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    # -------------------------------------------------------------
    # TEMPORAL SPLITTING: Draw the line 90 days before the last date
    # -------------------------------------------------------------
    latest_date = df['InvoiceDate'].max()
    cutoff_date = latest_date - dt.timedelta(days=90)
    
    df_past = df[df['InvoiceDate'] <= cutoff_date]
    df_future = df[df['InvoiceDate'] > cutoff_date]
    
    print("\nCalculating RFM metrics using ONLY past data...")
    rfm = df_past.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (cutoff_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })
    
    rfm.rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)
    
    # Create RFM Scores
    r_labels, f_labels, m_labels = range(4, 0, -1), range(1, 5), range(1, 5)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=f_labels)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    # -------------------------------------------------------------
    # CUSTOMER SEGMENTATION
    # -------------------------------------------------------------
    def segment_customer(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 3 and f <= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 1 and f <= 1:
            return 'Lost'
        else:
            return 'Other'
            
    rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)
    
    # -------------------------------------------------------------
    # CALCULATE TRUE CHURN LABELS
    # -------------------------------------------------------------
    retained_customers = df_future['Customer ID'].unique()
    rfm['Is_Churn'] = (~rfm.index.isin(retained_customers)).astype(int)
    
    print(f"\nSaving robust RFM data to {output_path}...")
    rfm.to_csv(output_path)
    print("Advanced RFM Analysis complete!")

if __name__ == "__main__":
    calculate_rfm()
