import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import config safely
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def clean_telecom_data():
    """Loads, cleans, validates, and saves the telecom dataset."""
    print(f"Loading raw data from: {config.RAW_DATA_PATH}")
    
    try:
        # 1. Load Data
        df = pd.read_csv(config.RAW_DATA_PATH)
        initial_shape = df.shape
        print(f"Initial shape: {initial_shape}")

        # 2. Handle Duplicates
        df = df.drop_duplicates()

        # 3. Type Conversions & Invalid Records
        # TotalCharges has empty spaces (" ") for customers with 0 tenure. 
        # errors='coerce' forces these invisible spaces into NaN (Not a Number) values.
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # 4. Handle Missing Values
        # Because we coerced spaces to NaN, we now have ~11 missing values.
        # Since 11 rows is < 0.1% of 7000 rows, dropping them is statistically safe.
        df = df.dropna(subset=['TotalCharges'])

        # 5. Type Conversion for Target Variable
        # ML models need numbers, not text.
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        # 6. Outlier Handling (Using 99th Percentile Capping)
        # We cap extreme MonthlyCharges so outliers don't skew our logistic regression/trees.
        upper_limit = df['MonthlyCharges'].quantile(0.99)
        df['MonthlyCharges'] = np.where(df['MonthlyCharges'] > upper_limit, upper_limit, df['MonthlyCharges'])

        # 7. Save Processed Data
        os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
        df.to_csv(config.PROCESSED_DATA_PATH, index=False)
        
        final_shape = df.shape
        print(f"Data cleaning complete! Final shape: {final_shape}")
        print(f"Processed file saved to: {config.PROCESSED_DATA_PATH}")
        
        return df

    except Exception as e:
        print(f"CRITICAL ERROR during data cleaning: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clean_telecom_data()