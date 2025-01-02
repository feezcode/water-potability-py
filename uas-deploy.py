import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Title and description
st.title("Aplikasi Prediksi Kualitas Air")
st.write("""
Aplikasi ini memungkinkan Anda untuk mengeksplorasi dataset kualitas air dan memprediksi apakah air layak diminum berdasarkan parameter yang dimasukkan.
""")

# Load dataset
file_path = 'water_potability.csv'  # Pastikan file ini ada di direktori yang sama
data = pd.read_csv(file_path)

# Data preprocessing
st.header("Eksplorasi Data")

# Menampilkan beberapa baris pertama
st.subheader("Data Awal")
st.dataframe(data.head())

# Menampilkan informasi dataset
st.subheader("Informasi Dataset")
st.write(f"Jumlah Baris dan Kolom: {data.shape}")
st.write(data.info())

# Menangani missing values
st.subheader("Menangani Missing Values")
missing_values = data.isnull().sum()
st.write("Jumlah Missing Values per Kolom:")
st.write(missing_values)

# Imputasi dengan mean
data.fillna(data.mean(), inplace=True)

# Visualisasi distribusi target
st.subheader("Distribusi Target (Potability)")
fig, ax = plt.subplots()
data['Potability'].value_counts().plot(kind='bar', color=['blue', 'orange'], alpha=0.7, ax=ax)
ax.set_title("Distribusi Potability")
ax.set_xlabel("Potability (0: Tidak Layak, 1: Layak)")
ax.set_ylabel("Jumlah")
st.pyplot(fig)

# Split data menjadi fitur dan target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standarisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
st.header("Evaluasi Model")
models = {
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc

    # Confusion Matrix
    st.subheader(f"Confusion Matrix: {model_name}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["Not Potable", "Potable"], yticklabels=["Not Potable", "Potable"], ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# Menampilkan akurasi model
st.subheader("Perbandingan Akurasi Model")
st.write(results)

# Prediction
st.header("Prediksi Kualitas Air")
st.write("Masukkan nilai parameter untuk memprediksi apakah air layak diminum:")

# Input user
pH = st.number_input("pH (0-14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
Hardness = st.number_input("Hardness", value=100.0, step=1.0)
Solids = st.number_input("Solids", value=10000.0, step=1.0)
Chloramines = st.number_input("Chloramines", value=7.0, step=0.1)
Sulfate = st.number_input("Sulfate", value=200.0, step=1.0)
Conductivity = st.number_input("Conductivity", value=300.0, step=1.0)
Organic_carbon = st.number_input("Organic Carbon", value=10.0, step=0.1)
Trihalomethanes = st.number_input("Trihalomethanes", value=80.0, step=1.0)
Turbidity = st.number_input("Turbidity", value=3.0, step=0.1)

# Pilihan model
selected_model = st.selectbox("Pilih Model untuk Prediksi", list(models.keys()))
model = models[selected_model]

# Tombol prediksi
if st.button("Prediksi"):
    input_features = np.array([[pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
    input_features_scaled = scaler.transform(input_features)
    prediction = model.predict(input_features_scaled)[0]

    if prediction == 1:
        st.success("Air tersebut diprediksi **Layak Diminum**.")
    else:
        st.error("Air tersebut diprediksi **Tidak Layak Diminum**.")
