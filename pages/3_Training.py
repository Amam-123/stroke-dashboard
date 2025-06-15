import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

st.title("3. Training Model")

# Pastikan nama file sesuai dengan file di GitHub kamu
df = pd.read_csv("healthcare-dataset-stroke-data.csv")  # atau sesuaikan jika sudah di-preprocessing

# Contoh pemilihan fitur
X = df[["age", "avg_glucose_level", "bmi"]]
y = df["stroke"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "model_stroke.pkl")

st.success("Model berhasil dilatih dan disimpan sebagai model_stroke.pkl")
