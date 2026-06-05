import pandas as pd
import datetime as dt
import os

def calculate_rfm(input_path, output_path):
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate Total Price for each transaction line
    df['TotalPrice'] = df['Quantity'] * df['Price']
    
    # Define a baseline date (e.g., the day after the last transaction in the dataset)
    latest_date = df['InvoiceDate'].max()
    baseline_date = latest_date + dt.timedelta(days=1)
    
    print("Calculating RFM metrics...")
    # Group by Customer ID
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (baseline_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    })
    
    # Rename columns
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    # Create RFM Scores (1-4 scale, 4 is best)
    # Recency: Lower is better, so labels are reversed
    r_labels = range(4, 0, -1)
    f_labels = range(1, 5)
    m_labels = range(1, 5)
    
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
    # Frequency can have many ties at 1, which breaks qcut. Use rank method.
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=f_labels)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)
    
    # Combine scores
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    # Create basic segments mapping
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
    
    print("Segment distribution:")
    print(rfm['Customer_Segment'].value_counts())
    
    print(f"Saving RFM data to {output_path}...")
    rfm.to_csv(output_path)
    print("RFM Analysis Done!")

if __name__ == "__main__":
    cleaned_file = os.path.join("data", "processed", "online_retail_cleaned.csv")
    rfm_file = os.path.join("data", "processed", "rfm_data.csv")
    calculate_rfm(cleaned_file, rfm_file)
