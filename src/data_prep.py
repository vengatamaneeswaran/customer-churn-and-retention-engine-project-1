import pandas as pd
import os

def clean_data():
    input_path = os.path.join("data", "raw", "online_retail_II.xlsx")
    output_path = os.path.join("data", "processed", "online_retail_cleaned.csv")
    
    print(f"Loading raw data from {input_path}...")
    # The dataset has multiple sheets; we will use the most recent year
    df = pd.read_excel(input_path, sheet_name="Year 2010-2011")
    print(f"Initial shape: {df.shape}")
    
    # 1. Drop rows with missing Customer ID
    df.dropna(subset=['Customer ID'], inplace=True)
    
    # 2. Filter out negative or zero quantities (returns/cancellations)
    df = df[df['Quantity'] > 0]
    
    # 3. Filter out negative or zero prices
    df = df[df['Price'] > 0]
    
    # 4. Remove explicitly canceled orders (Invoice number starts with 'C')
    # Since we already filtered Quantity > 0, cancellations are usually gone, 
    # but it's good practice to be explicit.
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    print(f"Cleaned shape: {df.shape}")
    print(f"Saving cleaned data to {output_path}...")
    
    df.to_csv(output_path, index=False)
    print("Data cleaning complete!")

if __name__ == "__main__":
    clean_data()
