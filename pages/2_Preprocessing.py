st.title("2. Preprocessing Data")

st.write("Langkah-langkah:")
st.markdown("""
- Menghapus missing value pada kolom `bmi`
- Encoding kolom `gender`, `smoking_status`, dst
""")

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.dropna()
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

st.dataframe(df.head())
