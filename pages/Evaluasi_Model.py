import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("4. Evaluasi Model")

# Load dataset hasil preprocessing
df = pd.read_csv("healthcare-dataset-stroke-data.csv")  # Pastikan nama dan lokasi file sesuai

# Pisahkan fitur dan label
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Load model
model = joblib.load("model_stroke.pkl")

# Prediksi
y_pred = model.predict(X)

# Evaluasi
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
st.pyplot(plt.gcf())
plt.clf()

st.subheader("Classification Report")
report = classification_report(y, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())
