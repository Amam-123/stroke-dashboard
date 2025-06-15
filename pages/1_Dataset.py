import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("1. Dataset dan Visualisasi Awal")

df = pd.read_csv("stroke.csv")  # sesuaikan nama file

st.write("### Contoh Data")
st.dataframe(df.head())

st.write("### Statistik Ringkas")
st.write(df.describe())

st.write("### Distribusi Umur")
fig, ax = plt.subplots()
sns.histplot(df['age'], kde=True, ax=ax)
st.pyplot(fig)
