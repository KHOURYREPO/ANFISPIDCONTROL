import pandas as pd
import numpy as np
import scipy
import matplotlib
import pysindy

print("All required packages imported successfully!")

# Define the data path
data_path = r"C:\Users\boutrous.khoury\anfisdatapid\data"

# Attempt to load the CSV files with additional parameters to handle parsing issues
try:
    # Reading with error handling for bad lines
    data1 = pd.read_csv(f"{data_path}\\tank_data_190625.csv", on_bad_lines='warn', delimiter=',')
    print("First data file loaded successfully from:", data_path)
    print("First few rows of data1:")
    print(data1.head())
    print("Columns in data1:", list(data1.columns))
except Exception as e:
    print("Error loading CSV files:", str(e))
    print("Attempting to load with alternative settings...")
    try:
        # Try loading with a different delimiter or skipping bad lines
        data1 = pd.read_csv(f"{data_path}\\tank_data_190625.csv", delimiter=',', on_bad_lines='skip')
        print("Loaded with skipped bad lines. First few rows:")
        print(data1.head())
        print("Columns in data1 after skipping bad lines:", list(data1.columns))
    except Exception as e2:
        print("Still failed to load CSV with alternative settings:", str(e2))
        exit()

# # Repeat for the second file if needed
# try:
#     data2 = pd.read_csv(f"{data_path}\\tank_data_180625.csv", on_bad_lines='warn', delimiter=',')
#     print("Second data file loaded successfully from:", data_path)
#     print("First few rows of data2:")
#     print(data2.head())
#     print("Columns in data2:", list(data2.columns))
# except Exception as e:
#     print("Error loading second CSV file:", str(e))
#     print("Proceeding with only the first file if loaded...")
#
# # If data1 is loaded, proceed with it for now
# if 'data1' in locals():
#     data = data1
#     print("Proceeding with first dataset only for now.")
# else:
#     print("No data loaded. Please check file paths and content.")
#     exit()
#
# # Further processing would go here...
# print("Data loading step completed. Check column names above to ensure correctness.")