import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to load config safely
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def engineer_features():
    """Generates 15 business-driven features and encodes categorical data."""
    print(f"Loading cleaned data from: {config.PROCESSED_DATA_PATH}")
    
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        
        # Protect against division by zero for tenure-based calculations
        df['tenure'] = df['tenure'].replace(0, 1) 
        
        # -----------------------------------------------------------
        # FEATURE 1-4: ENGAGEMENT & STICKINESS METRICS
        # -----------------------------------------------------------
        services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # 1. Total Services Count: How embedded are they in our ecosystem?
        df['Total_Services_Count'] = df[services].apply(lambda x: (x != 'No').sum(), axis=1)
        
        # 2. Security Services Count: Security users rarely churn.
        sec_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        df['Security_Services_Count'] = df[sec_services].apply(lambda x: (x == 'Yes').sum(), axis=1)
        
        # 3. Streaming Services Count: Entertainment users.
        df['Streaming_Services_Count'] = df[['StreamingTV', 'StreamingMovies']].apply(lambda x: (x == 'Yes').sum(), axis=1)
        
        # 4. Single Service User (Binary): High flight risk.
        df['Single_Service_User'] = np.where(df['Total_Services_Count'] == 1, 1, 0)

        # -----------------------------------------------------------
        # FEATURE 5-8: FINANCIAL & REVENUE METRICS
        # -----------------------------------------------------------
        # 5. Average Historical Monthly Charge
        df['Avg_Historical_Monthly'] = df['TotalCharges'] / df['tenure']
        
        # 6. Bill Creep (Current Monthly vs Historical)
        # If this is positive, their bill has gone up over time.
        df['Bill_Creep'] = df['MonthlyCharges'] - df['Avg_Historical_Monthly']
        
        # 7. Charge Per Service: Are they getting a good deal?
        df['Charge_Per_Service'] = df['MonthlyCharges'] / df['Total_Services_Count'].clip(lower=1)
        
        # 8. Is Auto Pay (Binary): Auto-payers "set it and forget it" (Lower churn).
        df['Is_Auto_Pay'] = df['PaymentMethod'].apply(lambda x: 1 if 'automatic' in x.lower() else 0)

        # -----------------------------------------------------------
        # FEATURE 9-12: DEMOGRAPHIC & FAMILY METRICS
        # -----------------------------------------------------------
        # 9. Family Bundle (Binary): Has both partner and dependents. Very stable.
        df['Family_Bundle'] = np.where((df['Partner'] == 'Yes') & (df['Dependents'] == 'Yes'), 1, 0)
        
        # 10. Senior with Dependents: Specific demographic slice.
        df['Senior_with_Dependents'] = np.where((df['SeniorCitizen'] == 1) & (df['Dependents'] == 'Yes'), 1, 0)
        
        # 11. Tenure Group: Binning continuous data into marketing segments.
        df['Tenure_Group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 100], 
                                    labels=['0-1_Year', '1-2_Years', '2-4_Years', '4-5_Years', '5+_Years'])
        
        # 12. High Risk Contract (Binary): Directly targets month-to-month.
        df['High_Risk_Contract'] = np.where(df['Contract'] == 'Month-to-month', 1, 0)

        # -----------------------------------------------------------
        # FEATURE 13-15: HYBRID RISK SCORES (A Priori Logic)
        # -----------------------------------------------------------
        # 13. Contract to Tenure Ratio
        # Maps contract lengths to months, then divides tenure by it to find "Renewal Cycles".
        contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        df['Contract_Length_Months'] = df['Contract'].map(contract_map)
        df['Tenure_to_Contract_Ratio'] = df['tenure'] / df['Contract_Length_Months']
        
        # 14. High Value, High Risk (Binary): The most important people to call.
        df['High_Value_High_Risk'] = np.where((df['MonthlyCharges'] > 70) & (df['High_Risk_Contract'] == 1), 1, 0)
        
        # 15. Loyalty Discount Eligible: Survived Year 1 but still on Month-to-Month.
        df['Loyalty_Discount_Eligible'] = np.where((df['tenure'] > 12) & (df['High_Risk_Contract'] == 1), 1, 0)

        # -----------------------------------------------------------
        # DATA ENCODING FOR MACHINE LEARNING
        # -----------------------------------------------------------
        # Drop columns that are useless for math (customerID) or redundant now (Contract_Length_Months)
        # We save customerID to a separate variable just in case we need it for the database later.
        customer_ids = df['customerID']
        df = df.drop(columns=['customerID', 'Contract_Length_Months'])
        
        # One-Hot Encoding: Converts text categories into binary math columns (e.g., Internet_Fiber_optic = 1 or 0)
        # drop_first=True avoids the "Dummy Variable Trap" (Multicollinearity)
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Put customerID back as the first column
        df_encoded.insert(0, 'customerID', customer_ids)
        
        # Save the engineered dataset
        df_encoded.to_csv(config.ENGINEERED_DATA_PATH, index=False)
        
        print(f"Feature Engineering complete! New shape: {df_encoded.shape}")
        print(f"Engineered data saved to: {config.ENGINEERED_DATA_PATH}")
        
        return df_encoded

    except Exception as e:
        print(f"Error during feature engineering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    engineer_features()