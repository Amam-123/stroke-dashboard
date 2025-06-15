from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

import streamlit as st
st.title("3. Pelatihan Model")

df = pd.read_csv("healthcare-dataset-stroke-data.csv")  # setelah preprocessing
X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
st.write("Akurasi Model:", acc)

joblib.dump(model, "model.pkl")
