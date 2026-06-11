import requests
import os

def download_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
    filepath = os.path.join("data", "raw", "online_retail_II.xlsx")
    
    print(f"Downloading dataset to {filepath}...")
    
    # Use requests to stream the large file reliably
    response = requests.get(url, stream=True)
    response.raise_for_status() # Check for any errors
    
    # This will overwrite the broken partial file from the previous attempt
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print("Download complete!")

if __name__ == "__main__":
    download_dataset()
