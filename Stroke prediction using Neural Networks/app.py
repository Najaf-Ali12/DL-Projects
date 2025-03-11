import streamlit as st
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.models import load_model
import joblib

import os

# List all files and folders in the current directory
files = os.listdir(os.getcwd())

print("Files and Folders in Current Directory:", files)

# Load your trained model (update the path to your model if needed)
model = load_model("models/stroke_prediction_model.h5")

# Load the trained scaler
with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

# Giving developer name
st.sidebar.markdown("Developer: Najaf Ali")

# Page Title
st.title("🩺 Stroke Prediction App")

# Add an image ()
image = Image.open("stroke prediction image.jpeg")  # Make sure the image is in the same directory
st.image(image, caption="Stroke Awareness", use_container_width=False)

st.write("### Answer the following questions to predict the chances of stroke:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Please enter your age",min_value=0,max_value=120)
if age>120:
    st.warning("As a model, I don't think a person have age more than 120 nowadays")
hypertension = st.selectbox("Do you have Hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Do you have any Heart Disease?", ["No", "Yes"])
ever_married = st.selectbox("Have you ever been married?", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
smoking_status = st.selectbox("Smoking Status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])

# Convert inputs into numeric format for prediction
if st.button("Predict Stroke"):
    gender = 1 if gender == "Male" else (2 if gender == "Other" else 0)
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    work_type = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "Children": 4, "Never_worked": 1}[work_type]
    Residence_type = 1 if Residence_type == "Urban" else 0
    smoking_status = {"Never smoked": 2, "Formerly smoked": 1, "Smokes": 3, "Unknown": 0}[smoking_status]

    features = np.array([[gender, age, hypertension, heart_disease, ever_married,
                          work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])
    
    #Scaling the input data to get correct predictions
    inputs_to_scaled=[age,bmi,avg_glucose_level]
    scaler.transform(np.array(inputs_to_scaled).reshape(1,-1))
    
    #Prediction ()
    prediction = model.predict(features)
    result = "Stroke Risk" if prediction[0] == 1 else "No Stroke Risk"
    
    # For now, just show entered data as confirmation
    st.success(f"Your data has been submitted for prediction.")
    st.success(f"Prediction: {result}")


# ⭐ User Review Section
st.subheader("⭐ Rate This Project")
rating=st.feedback("stars")
st.write(f"Thank you for your feedback! 🌟")

# 💬 User Comment Section
st.subheader("💬 Share Your Thoughts")
comment = st.text_area("Leave a comment below:")
if st.button("Submit Feedback"):
    st.success("Thank you for your valuable feedback! 😊")
