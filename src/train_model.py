import pandas as pd
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add parent directory to path to load config safely
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def train_and_evaluate():
    """Trains a Random Forest model, evaluates business metrics, and saves the model."""
    print(f"Loading engineered data from: {config.ENGINEERED_DATA_PATH}")
    
    try:
        # 1. Load Data
        df = pd.read_csv(config.ENGINEERED_DATA_PATH)
        
        # 2. Separate Features (X) and Target (y)
        # We drop customerID because it's an arbitrary string that confuses the math.
        X = df.drop(columns=['customerID', 'Churn'])
        y = df['Churn']
        
        # 3. Train-Test Split (80% for training, 20% for testing)
        # stratify=y ensures the 74/26 Yes/No ratio is maintained in both splits.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training Data Shape: X={X_train.shape}, y={y_train.shape}")
        
        # 4. Initialize the Algorithm
        # class_weight='balanced' is CRITICAL for imbalanced datasets.
        # It forces the model to pay heavy attention to minority class (Churners).
        model = RandomForestClassifier(
            n_estimators=100,      # Number of trees in the forest
            max_depth=10,          # Maximum depth of each tree (prevents overfitting)
            min_samples_split=5,   # Minimum samples required to split a node
            class_weight='balanced', 
            random_state=42
        )
        
        # 5. Train the Model
        print("\nTraining the Random Forest model... This may take a few seconds.")
        model.fit(X_train, y_train)
        
        # 6. Make Predictions on the Test Set
        # predict() gives 0 or 1. predict_proba() gives the exact percentage risk (e.g., 0.82)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] 
        
        # 7. Evaluate the Model (Phase 7)
        print("\n" + "="*40)
        print("MODEL EVALUATION METRICS")
        print("="*40)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        # 8. Feature Importance (What is driving churn?)
        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("\nTop 5 Churn Drivers:")
        print(feature_importances.head(5).to_string(index=False))

        # 9. Save the Model
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        joblib.dump(model, config.MODEL_PATH)
        print(f"\nModel successfully saved to: {config.MODEL_PATH}")

        # Optional: Save a visual Confusion Matrix for your GitHub / Portfolio
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(config.BASE_DIR, "notebooks", "confusion_matrix.png"))
        
        return model

    except Exception as e:
        print(f"CRITICAL ERROR during model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    train_and_evaluate()