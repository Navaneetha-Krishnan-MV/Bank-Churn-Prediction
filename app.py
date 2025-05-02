import streamlit as st
import pandas as pd
import requests

st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Predict if a customer will leave the bank")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        age = st.slider("Age", 18, 100, 35)
        balance = st.number_input("Account Balance ($)", 0.0, 500000.0, 1000.0)
        products = st.selectbox("Number of Products", [1, 2, 3, 4])

    with col2:
        tenure = st.slider("Tenure (years)", 0, 10, 2)
        has_card = st.checkbox("Has Credit Card")
        is_active = st.checkbox("Is Active Member")
        salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0)
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.radio("Gender", ["Male", "Female"])

    submitted = st.form_submit_button("Predict Churn")

# Process input and call API
if submitted:
    # Prepare JSON payload
    payload = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": int(has_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": salary
    }

    try:
        # Call Flask API
        response = requests.post("http://localhost:5000/predict", json=payload)
        result = response.json()

        # Show result
        probability = result['probability']
        prediction = result['prediction']

        st.subheader("Prediction Result")
        if probability > 0.6:
          st.error(f"🚨 Very High churn risk! (Probability: {probability:.2%})")
        elif probability > 0.4:
          st.warning(f"⚠️ Moderate churn risk (Probability: {probability:.2%})")
        elif probability > 0.25:
          st.info(f"ℹ️ Slight churn risk (Probability: {probability:.2%})")
        else:
          st.success(f"✅ Low churn risk (Probability: {probability:.2%})")

    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
