import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def train_churn_model(rfm_path, clv_path, model_path):
    print("Loading data...")
    rfm = pd.read_csv(rfm_path)
    clv = pd.read_csv(clv_path)
    
    # Merge datasets
    df = pd.merge(rfm, clv, on='Customer ID', how='inner')
    
    # We define Churn as a customer who hasn't made a purchase in the last 90 days
    # (i.e. Recency > 90). This is a standard retail churn definition.
    df['Is_Churn'] = (df['Recency'] > 90).astype(int)
    
    print(f"Target Distribution:\n{df['Is_Churn'].value_counts(normalize=True) * 100}")
    
    # Select features. We omit 'Recency' because the target is perfectly derived from it.
    # We use spending habits (Monetary), purchase counts (Frequency), and advanced CLV metrics.
    features = ['Frequency', 'Monetary', 'predicted_monetary_value', 'CLV_12m', 'RFM_Score']
    
    X = df[features].fillna(0) # Fill NaN from customers with only 1 purchase
    y = df['Is_Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Feature Importance
    importance = xgb_model.feature_importances_
    print("Feature Importances:")
    for feat, imp in zip(features, importance):
        print(f" - {feat}: {imp:.4f}")
        
    print(f"\nSaving model to {model_path}...")
    xgb_model.save_model(model_path)
    print("Done!")

if __name__ == "__main__":
    rfm_file = os.path.join("data", "processed", "rfm_data.csv")
    clv_file = os.path.join("data", "processed", "clv_data.csv")
    model_output = os.path.join("models", "xgboost_churn_model.json")
    train_churn_model(rfm_file, clv_file, model_output)
