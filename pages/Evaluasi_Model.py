import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("4. Evaluasi Model")

# Load data hasil split & model
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")
model = joblib.load("model_stroke.pkl")

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)
