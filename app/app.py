import streamlit as st
import pandas as pd

from utils import (
    predict_customer,
    risk_segment,
    value_segment,
    priority_segment,
    retention_action,
    get_shap_values,
    get_top_features,
    generate_explanation
)


# PAGE CONFIG
st.set_page_config(
    page_title="Customer Churn Intelligence System",
    layout="wide"
)

st.title("Customer Churn Intelligence System")
# SIDEBAR (GEMINI KEY
st.sidebar.header("Gemini Settings")

api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    [
        "gemini-flash-latest",   # BEST (works)
        "gemini-1.0-pro"         # fallback
    ]
)
# INPUT FOR
st.subheader("Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 20, 85, 30)

with col2:
    driving_license = st.selectbox("Driving License", [0, 1])
    previously_insured = st.selectbox("Previously Insured", [0, 1])

with col3:
    vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
    vehicle_damage = st.selectbox("Vehicle Damage", ["Yes", "No"])

annual_premium = st.number_input("Annual Premium", 2000, 100000, 30000)
policy_channel = st.number_input("Policy Sales Channel", 1, 200, 26)
vintage = st.slider("Customer Vintage", 10, 300, 150)
region_code = st.number_input("Region Code", 0, 52, 28)
# CREATE INPUT D
input_data = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Driving_License": driving_license,
    "Region_Code": region_code,
    "Previously_Insured": previously_insured,
    "Vehicle_Age": vehicle_age,
    "Vehicle_Damage": vehicle_damage,
    "Annual_Premium": annual_premium,
    "Policy_Sales_Channel": policy_channel,
    "Vintage": vintage
}])
# PREDICTION BUTTO
if st.button("Predict Churn"):

    prob, pred = predict_customer(input_data)

    risk = risk_segment(prob)
    value = value_segment(annual_premium)
    priority = priority_segment(risk, value)
    action = retention_action(priority)

    # DISPLAY RESULTS
    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Churn Probability", f"{prob:.2f}")
    col2.metric("Risk Category", risk)
    col3.metric("Priority", priority)

    st.success(f"Recommended Action: {action}")

    # SHAP EXPLANATION
    st.subheader("Key Factors (Model Explanation)")

    shap_vals, X = get_shap_values(input_data)
    top_features = get_top_features(shap_vals[0], X.iloc[0])

    for feat, val in top_features:
        st.write(f"{feat} (impact: {val:.2f})")

    # GEMINI EXPLANATION
    st.subheader("AI Explanation")

    if not api_key:
        st.warning("Please enter your Gemini API key to enable AI explanation.")
    else:
        explanation = generate_explanation(
            api_key,
            risk,
            priority,
            top_features,
            model_choice
        )
        
        st.write(explanation)