import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def train_churn_model():
    rfm_path = os.path.join("data", "processed", "rfm_data.csv")
    clv_path = os.path.join("data", "processed", "clv_data.csv")
    model_path = os.path.join("models", "xgboost_churn_model.json")
    
    print("Loading RFM and CLV data...")
    rfm = pd.read_csv(rfm_path)
    clv = pd.read_csv(clv_path)
    
    # Merge datasets
    df = pd.merge(rfm, clv, on='Customer ID', how='inner')
    
    # Define Target: 1 if they haven't purchased in >90 days, else 0
    df['Is_Churn'] = (df['Recency'] > 90).astype(int)
    
    print(f"\nTarget Distribution (Churned vs Retained):\n{df['Is_Churn'].value_counts(normalize=True) * 100}")
    
    # Select features
    # We omit 'Recency' because the target is perfectly derived from it.
    features = ['Frequency', 'Monetary', 'predicted_monetary_value', 'CLV_12m', 'RFM_Score']
    
    X = df[features].fillna(0) # Fill missing values for customers with only 1 purchase
    y = df['Is_Churn']
    
    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTraining XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    print("\nEvaluating Model...")
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Feature Importance
    print("Feature Importances:")
    importance = xgb_model.feature_importances_
    for feat, imp in zip(features, importance):
        print(f" - {feat}: {imp:.4f}")
        
    print(f"\nSaving model to {model_path}...")
    xgb_model.save_model(model_path)
    print("Machine Learning Pipeline complete!")

if __name__ == "__main__":
    train_churn_model()
