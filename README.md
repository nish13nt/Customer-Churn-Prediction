# Customer Churn Prediction & Intelligence Platform

An end-to-end Machine Learning and Business Intelligence (BI) platform that predicts telecommunication customer defection. The platform automates data sanitation, establishes a normalized local Data Warehouse, builds an optimized Random Forest classifier, exposes real-time predictive capabilities via a RESTful FastAPI microservice, and delivers interactive analytics through Power BI.

---

## 🏗️ System Architecture & Data Flow

1. **Data Ingestion:** Reads the raw historical customer dataset (`.csv`).
2. **Data Sanitation Layer:** Handles data type anomalies, manages missing financial variables, caps outliers, and maps categorical indicators.
3. **Data Warehouse Layer:** Automatically creates a normalized 3NF MySQL database schema with optimized indexing structures.
4. **Feature Engineering Engine:** Formulates 15 custom business metrics evaluating customer product engagement, price sensitivity, and billing adjustments.
5. **Machine Learning Core:** Trains and saves an imbalanced-class adjusted Random Forest model optimizing for target Recall (82%).
6. **API Microservice:** Exposes real-time model inference and rule-based business recommendations via FastAPI routes.
7. **Business Intelligence Layer:** Connects processed data assets to Power BI dashboards utilizing custom DAX modeling to map financial revenue at risk.

---

## 📂 Project Repository Structure

```text
customer-churn-platform/
├── data/
│   ├── raw/
│   │   └── telecom_churn.csv          # Immutable source data
│   └── processed/
│       ├── cleaned_data.csv          # Sanitized output
│       └── engineered_data.csv       # 44-column ML ready matrix
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   └── confusion_matrix.png          # Saved model performance graphic
├── src/
│   ├── __init__.py
│   ├── database.py                   # MySQL schema deployment script
│   ├── data_cleaning.py              # Inversion and null-handling logic
│   ├── feature_engineering.py       # 15 Custom DAX-equivalent feature builders
│   ├── train_model.py                # Model generation and assessment script
│   └── predict.py                    # Service layer mapping business logic
├── api/
│   ├── __init__.py
│   └── main.py                       # FastAPI application and routes
├── models/
│   └── churn_model.pkl               # Saved serialization of the trained model
├── dashboard/
│   └── churn_analysis.pbix           # Completed Power BI workspace report
├── .env                              # Secured local credentials (git ignored)
├── .gitignore                        # Git structural upload filter
├── config.py                         # Single source of truth settings architecture
└── requirements.txt                  # Python dependency catalog
