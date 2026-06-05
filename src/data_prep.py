import pandas as pd
import os

def clean_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    # The dataset has multiple sheets, but typically we use the first one or combine them
    # "Year 2009-2010" and "Year 2010-2011"
    # Let's load the 2010-2011 sheet for the most recent data
    df = pd.read_excel(input_path, sheet_name="Year 2010-2011")
    
    print(f"Initial shape: {df.shape}")
    
    # 1. Drop rows with missing Customer ID
    df.dropna(subset=['Customer ID'], inplace=True)
    
    # 2. Filter out negative quantities (returns/cancellations)
    df = df[df['Quantity'] > 0]
    
    # 3. Filter out negative or zero prices
    df = df[df['Price'] > 0]
    
    # 4. Remove canceled orders (Invoice number starts with 'C')
    # Since we already filtered Quantity > 0, cancellations are usually gone, 
    # but let's be explicit.
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    print(f"Cleaned shape: {df.shape}")
    
    print(f"Saving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    raw_file = os.path.join("data", "raw", "online_retail_II.xlsx")
    processed_file = os.path.join("data", "processed", "online_retail_cleaned.csv")
    clean_data(raw_file, processed_file)
