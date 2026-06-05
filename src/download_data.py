import urllib.request
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
filepath = os.path.join("data", "raw", "online_retail_II.xlsx")

if not os.path.exists(filepath):
    print(f"Downloading {url} to {filepath}...")
    urllib.request.urlretrieve(url, filepath)
    print("Download complete.")
else:
    print("File already exists. Skipping download.")
