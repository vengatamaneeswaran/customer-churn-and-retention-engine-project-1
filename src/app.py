import streamlit as st
import pandas as pd
import xgboost as xgb
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Customer Retention Engine", layout="wide")
st.title("🛍️ Customer Churn & Retention Engine")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    rfm_path = os.path.join("data", "processed", "rfm_data.csv")
    clv_path = os.path.join("data", "processed", "clv_data.csv")
    
    rfm = pd.read_csv(rfm_path)
    clv = pd.read_csv(clv_path)
    df = pd.merge(rfm, clv, on='Customer ID', how='inner')
    return df

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "xgboost_churn_model.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

df = load_data()
model = load_model()

# --- SIDEBAR ---
st.sidebar.header("Search Customer")
customer_ids = df['Customer ID'].astype(str).tolist()
selected_customer_str = st.sidebar.selectbox("Select or Type Customer ID", ["None"] + customer_ids)

# --- MACRO VIEW ---
if selected_customer_str == "None":
    st.subheader("Business Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Total Historical Revenue", f"${df['Monetary'].sum():,.2f}")
    col3.metric("Average 12M CLV", f"${df['CLV_12m'].mean():,.2f}")
    
    churn_rate = df['Is_Churn'].mean() * 100
    col4.metric("Actual Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("---")
    st.subheader("Customer Segments")
    st.bar_chart(df['Customer_Segment'].value_counts())
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(50))

# --- MICRO VIEW (Specific Customer) ---
else:
    selected_customer = float(selected_customer_str)
    cust_data = df[df['Customer ID'] == selected_customer].iloc[0]
    
    st.subheader(f"Customer Profile: {selected_customer_str}")
    
    # Predict live Churn Probability using the XGBoost Model!
    features = ['Frequency', 'Monetary', 'predicted_monetary_value', 'CLV_12m', 'RFM_Score']
    X_single = cust_data[features].to_frame().T.astype(float)
    churn_prob = float(model.predict_proba(X_single)[0][1] * 100)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📊 RFM Metrics")
        st.write(f"**Segment:** {cust_data['Customer_Segment']}")
        st.write(f"**Purchases (Frequency):** {int(cust_data['Frequency'])}")
        st.write(f"**Total Spent (Monetary):** ${cust_data['Monetary']:,.2f}")
        st.write(f"**RFM Score:** {cust_data['RFM_Score']}")
        
    with col2:
        st.success("💰 Lifetime Value Predictions")
        st.write(f"**Predicted Avg Order Value:** ${cust_data['predicted_monetary_value']:,.2f}")
        st.write(f"**Expected 12-Month CLV:** ${cust_data['CLV_12m']:,.2f}")
        st.write(f"**Probability of Being Alive:** {cust_data['p_alive']*100:.1f}%")
        
    with col3:
        st.error("⚠️ Churn Risk")
        st.metric("Live AI Churn Probability", f"{churn_prob:.1f}%")
        st.progress(churn_prob / 100)
        
        if churn_prob > 70:
            st.warning("High Risk of Churn! Recommend targeted discount or check-in email.")
        elif churn_prob < 30:
            st.balloons()
            st.success("Healthy Customer. Recommend loyalty rewards.")
        else:
            st.info("Moderate Risk. Monitor activity.")
