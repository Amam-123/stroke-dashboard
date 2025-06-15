import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Stroke Prediction", layout="centered")
st.title("Stroke Prediction Dashboard")

# Form input pengguna
st.subheader("Masukkan Data Pasien")

id_val = st.number_input("ID Pasien", min_value=0)

age = st.number_input("Usia (tahun)", min_value=0.0)
hypertension = st.selectbox("Hipertensi", [0, 1])
heart_disease = st.selectbox("Penyakit Jantung", [0, 1])
avg_glucose_level = st.number_input("Rata-rata Glukosa", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)

gender = st.selectbox("Jenis Kelamin", ["Male", "Other", "Female"])
ever_married = st.selectbox("Pernah Menikah", ["Yes", "No"])
work_type = st.selectbox("Tipe Pekerjaan", ["Private", "Self-employed", "children", "Never_worked"])
residence_type = st.selectbox("Tempat Tinggal", ["Urban", "Rural"])
smoking_status = st.selectbox("Status Merokok", ["never smoked", "formerly smoked", "smokes"])

# One-hot encoding manual (harus sesuai urutan kolom training)
gender_Male = 1 if gender == "Male" else 0
gender_Other = 1 if gender == "Other" else 0
ever_married_Yes = 1 if ever_married == "Yes" else 0

work_type_Never_worked = 1 if work_type == "Never_worked" else 0
work_type_Private = 1 if work_type == "Private" else 0
work_type_Self_employed = 1 if work_type == "Self-employed" else 0
work_type_children = 1 if work_type == "children" else 0

Residence_type_Urban = 1 if residence_type == "Urban" else 0

smoking_formerly = 1 if smoking_status == "formerly smoked" else 0
smoking_never = 1 if smoking_status == "never smoked" else 0
smoking_smokes = 1 if smoking_status == "smokes" else 0

# Susun input sesuai urutan kolom saat training
input_data = [[
    id_val,
    age,
    hypertension,
    heart_disease,
    avg_glucose_level,
    bmi,
    gender_Male,
    gender_Other,
    ever_married_Yes,
    work_type_Never_worked,
    work_type_Private,
    work_type_Self_employed,
    work_type_children,
    Residence_type_Urban,
    smoking_formerly,
    smoking_never,
    smoking_smokes
]]

# Tombol prediksi
if st.button("Prediksi Stroke"):
    prediction = model.predict(input_data)
    hasil = "POTENSI STROKE" if prediction[0] == 1 else "TIDAK ADA STROKE"
    st.success(f"Hasil Prediksi: {hasil}")
