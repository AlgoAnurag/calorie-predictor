import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('calories_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title and description
st.title("🔥 Calories Burnt Prediction 🔥")
st.markdown("""
### Know how much energy you're burning during exercise! 🏋️‍♂️🏃‍♀️
Fill in the details below to get an estimate of the calories burnt.
""")

# Create columns for better layout
col1, col2 = st.columns(2)

# Inputs
with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)

with col2:
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=200.0, value=70.0)
    duration = st.number_input("Duration of Exercise (minutes)", min_value=0.0, max_value=300.0, value=30.0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=200.0, value=80.0)

body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)

# Add a nice button
if st.button("💪 Predict Calories Burnt 💪"):
    # Encode gender
    gender_encoded = 1 if gender == "Male" else 0
    
    # Prepare input data
    input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    prediction = model.predict(input_data)
    
    # Display result with some styling
    st.success(f"🔥 Estimated Calories Burnt: **{prediction[0]:.2f}** 🔥")
