import streamlit as st
import pandas as pd
import joblib

st.title("Stroke Prediction App")
model = joblib.load("model.pkl")

# INPUT USER
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=0)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
avg_glucose = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)

# Ubah input categorical ke bentuk sesuai model
gender = 1 if gender == 'Male' else 0

# Prediksi
if st.button("Predict"):
    input_data = [[gender, age, hypertension, heart_disease, avg_glucose, bmi]]
    prediction = model.predict(input_data)
    st.success(f"Prediction: {'STROKE' if prediction[0]==1 else 'NO STROKE'}")
