import streamlit as st
import pandas as pd

st.title("2. Preprocessing Data")

# Contoh isi awal, nanti bisa kamu lengkapi:
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Preprocessing sederhana
st.write("Jumlah data null:")
st.write(df.isnull().sum())

st.write("Dataset setelah drop NA:")
df_clean = df.dropna()
st.dataframe(df_clean)
