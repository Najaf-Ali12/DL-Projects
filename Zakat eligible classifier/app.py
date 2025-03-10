import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import joblib
from PIL import Image

# Load the scaler and model with error handling
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = load_model("Zakat eligibility predictor.keras", compile=False)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Page title
st.title("📢 Zakat Eligibility Detector 📢")

# Add an image
try:
    image = Image.open("zakat image.jpeg")
    st.image(image, caption="Zakat Eligibility Detector", use_container_width=True)
except FileNotFoundError:
    st.warning("Zakat image not found. Please ensure 'zakat image.jpeg' exists in the directory.")

# Getting input from user
st.text("Please provide the following information to determine Zakat eligibility.")

name = st.text_input("Enter the name:", key='name')
gold = st.number_input("Enter the grams of gold:", min_value=0.0, key='gold')
m_income = st.number_input("Enter monthly income in PKR:", min_value=0.0, key='m_income')
m_expense = st.number_input("Enter monthly expenses in PKR:", min_value=0.0, key='m_expenses')
income = st.number_input("Enter annual income in USD:", min_value=0.0, key='income')
assets = st.number_input("Enter value of assets in USD:", min_value=0.0, key='assets')
liability = st.number_input("Enter liabilities in USD:", min_value=0.0, key='liabilities')
dependents = st.number_input("Enter number of dependents:", min_value=0, step=1, key='dependents')
net_worth = st.number_input("Enter net worth in USD:", key='networth')

marital_status = st.selectbox("Select marital status:", options=['Single', 'Married', 'Divorced', 'Widowed'], key='marital_status')
employment_type = st.selectbox("Select employment type:", options=['Unemployed', 'Freelancer', 'Salaried', 'Business'], key='employment')
region = st.selectbox("Select the region:", options=['Rural', 'Urban'], key='region')

# Convert categorical values to numerical values
marital_status_dict = {'Single': 2, 'Married': 1, 'Divorced': 0, 'Widowed': 3}
employment_type_dict = {'Unemployed': 3, 'Freelancer': 1, 'Salaried': 2, 'Business': 0}
region_dict = {'Rural': 0, 'Urban': 1}

marital_status = marital_status_dict.get(marital_status, 2)
employment_type = employment_type_dict.get(employment_type, 3)
region = region_dict.get(region, 0)

# Predict eligibility when the button is clicked
if st.button("Check Eligibility"):
    try:
        # Prepare input data
        inputs_to_be_scalled=[gold,m_income,m_expense,income,assets,liability,net_worth] 
        inputs_scaled = scaler.transform(np.array(inputs_to_be_scalled).reshape(1, -1))

        input_data = np.array([[gold, m_income, m_expense, income, assets, liability, dependents, net_worth, marital_status, employment_type, region]])

        # Get prediction    
        prediction = model.predict(input_data)

        # Display result
        if prediction[0][0] > 0.5:
            st.success(f"{name} is **eligible** to receive Zakat. ✅")
        else:
            st.error(f"{name} is **not eligible** to receive Zakat. ❌")

    except Exception as e:
        st.error(f"An error occurred while processing: {e}")
