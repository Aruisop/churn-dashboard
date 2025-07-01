import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(layout="wide")
st.title("AI-Powered Customer Churn Prediction Dashboard")

# Load model
model = joblib.load("model/churn_model.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    orig_data = data.copy()
    
# Drop target label if it exists
if 'Churn' in data.columns:
    data = data.drop('Churn', axis=1)

    # Factorize categorical variables
    for col in data.select_dtypes(include="object").columns:
        data[col] = pd.factorize(data[col])[0]
    
    predictions = model.predict_proba(data)[:, 1]
    orig_data["Churn Probability"] = predictions

    st.subheader("Top At-Risk Customers")
    st.dataframe(orig_data.sort_values("Churn Probability", ascending=False).head(10))

    fig = px.histogram(orig_data, x="Churn Probability", nbins=20, title="Churn Probability Distribution")
    st.plotly_chart(fig, use_container_width=True)
