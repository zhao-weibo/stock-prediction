import pandas as pd
import os

data_folder = "data/full_history"
file_name = "A.csv"
file_path = os.path.join(data_folder, file_name)

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Read the CSV file
    try:
        data = pd.read_csv(file_path)
        
        # Display the first few rows of the dataset
        print("First 5 rows of the dataset:")
        print(data.head())

        print("Last 5 rows of the dataset:")
        print(data.tail())

        # Basic information of the dataset
        print("\nDataset Info:")
        print(data.info())

        # Check for missing values
        print("\nMissing values in each column:")
        print(data.isnull().sum())

        # Display basic statistical summary
        print("\nBasic statistical summary:")
        print(data.describe(include='all'))
    except Exception as e:
        print(f"Error reading the file: {e}")