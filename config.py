import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Database Configurations
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "churn_platform")

# Project Paths (Windows safe paths using os.path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "telecom_churn.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
ENGINEERED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "engineered_data.csv")