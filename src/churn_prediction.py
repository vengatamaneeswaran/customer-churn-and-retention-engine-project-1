import pandas as pd
import os
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def train_churn_model():
    rfm_path = os.path.join("data", "processed", "rfm_data.csv")
    clv_path = os.path.join("data", "processed", "clv_data.csv")
    
    print("Loading data...")
    rfm = pd.read_csv(rfm_path)
    clv = pd.read_csv(clv_path)
    
    # Merge datasets
    df = pd.merge(rfm, clv, on='Customer ID', how='inner')
    
    # We already calculated the true Is_Churn in rfm_analysis.py
    y = df['Is_Churn']
    
    # We use 'Frequency_x' which comes from the RFM dataframe
    features = ['Frequency', 'Monetary', 'predicted_monetary_value', 'CLV_12m', 'RFM_Score']
    X = df[features].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # CALCULATE CLASS IMBALANCE RATIO
    scale_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    print(f"Class Imbalance scale_pos_weight: {scale_weight:.2f}")
    
    print("Training Advanced XGBoost Classifier...")
    # Apply the scale_pos_weight so XGBoost pays more attention to churners
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_weight, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    print("\nEvaluating Model...")
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    print("\nGenerating SHAP Explanations...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    # Save SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    print("Saved SHAP Summary Plot to 'shap_summary_plot.png'")
    print("Machine Learning Pipeline complete!")

if __name__ == "__main__":
    train_churn_model()
