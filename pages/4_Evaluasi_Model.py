import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("4. Evaluasi Model")

# Load model dari folder utama
model = pickle.load(open("model_stroke.pkl", "rb"))

# Load data test
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
st.pyplot(fig)

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report)

st.subheader("Akurasi Model")
acc = accuracy_score(y_test, y_pred)
st.success(f"Akurasi: {acc:.2f}")
