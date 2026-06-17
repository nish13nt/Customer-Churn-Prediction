import joblib
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Load the model once globally so it doesn't reload on every single prediction (saves memory)
try:
    model = joblib.load(config.MODEL_PATH)
except Exception as e:
    print(f"Error loading model. Ensure Phase 6 was completed. {e}")

def generate_business_rules(probability: float, monthly_charges: float) -> dict:
    """
    Phase 8 & 9: Converts ML probability into business risk scores and actionable rules.
    """
    risk_score = int(probability * 100)
    
    # Business Logic Engine
    if risk_score >= 80:
        if monthly_charges > 70:
            action = "URGENT: High Value Customer. Manager Phone Call + Offer 20% Annual Discount."
        else:
            action = "HIGH RISK: Send Automated Email + Offer 10% Discount."
    elif risk_score >= 50:
        action = "MEDIUM RISK: Send 'Check-in' Survey and monitor usage."
    else:
        action = "LOW RISK: Standard Marketing Flow. No discount needed."
        
    return {
        "risk_score": risk_score,
        "recommendation": action
    }

def make_prediction(customer_features: dict) -> dict:
    """Receives a single customer's data, predicts churn, and attaches business rules."""
    # Convert incoming JSON dictionary to a Pandas DataFrame (1 row)
    df = pd.DataFrame([customer_features])
    
    # The model's predict_proba returns an array like [[0.15, 0.85]] (Stay vs Churn)
    # We want the second number (index 1), which is the probability of Churn.
    churn_probability = model.predict_proba(df)[0][1]
    
    # Extract monthly charges to feed into our business rules (default to 50 if missing)
    monthly_charges = customer_features.get('MonthlyCharges', 50.0)
    
    # Get business insights
    insights = generate_business_rules(churn_probability, monthly_charges)
    
    return {
        "churn_probability": round(churn_probability, 3),
        "risk_score": insights["risk_score"],
        "action_recommendation": insights["recommendation"]
    }