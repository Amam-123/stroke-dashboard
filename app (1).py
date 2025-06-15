import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Stroke Prediction", layout="centered")

# Judul halaman
st.title("Stroke Prediction Dashboard")

# Load model
model = joblib.load("model.pkl")

# Input user
age = st.number_input("Age", min_value=0)
avg_glucose = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)

# Prediksi
if st.button("Predict"):
    prediction = model.predict([[age, avg_glucose, bmi]])
    st.success(f"Prediction: {'STROKE' if prediction[0]==1 else 'NO STROKE'}")
