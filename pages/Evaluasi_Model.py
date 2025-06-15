import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

st.title("4. Evaluasi Model")

# Load data dan model
df = pd.read_csv("processed_stroke.csv")
X = df.drop("stroke", axis=1)
y = df["stroke"]

model = joblib.load("model.pkl")
y_pred = model.predict(X)

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")
report = classification_report(y, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# ROC Curve
st.subheader("ROC Curve")
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc_score = roc_auc_score(y, y_proba)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("Model tidak mendukung ROC Curve (predict_proba tidak tersedia).")
