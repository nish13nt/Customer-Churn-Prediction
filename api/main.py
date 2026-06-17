from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import make_prediction
from src.utils_logger import logger  # <--- IMPORT THE PRODUCTION LOGGER

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time ML inference and business recommendation engine.",
    version="1.0.0"
)

class CustomerPayload(BaseModel):
    features: dict

@app.get("/health")
def health_check():
    logger.info("Health check endpoint pinged.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_churn(payload: CustomerPayload):
    logger.info("Received real-time prediction request payload.")
    try:
        result = make_prediction(payload.features)
        logger.info(f"Prediction successful. Risk Score generated: {result['risk_score']}")
        return result
    except ValueError as ve:
        logger.warning(f"Data formatting validation failure: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"CRITICAL SYSTEM EXCEPTION during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")