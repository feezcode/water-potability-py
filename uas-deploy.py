import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
file_path = 'water_potability.csv'  # Adjust this path
data = pd.read_csv(file_path)

st.title("Water Potability Analysis")
st.write("This app analyzes the water potability dataset and compares multiple classification models.")

# Display dataset
st.header("Dataset Overview")
if st.checkbox("Show Raw Dataset"):
    st.write(data)

st.subheader("Basic Dataset Information")
st.write("Shape of dataset:", data.shape)
st.write("Missing values per column:")
st.write(data.isnull().sum())

# Handle missing values
data_imputed = data.fillna(data.mean())
st.write("Missing values after imputation:")
st.write(data_imputed.isnull().sum())

# Visualize distribution of target
st.subheader("Class Distribution")
fig, ax = plt.subplots()
data['Potability'].value_counts().plot(kind='bar', color=['blue', 'orange'], alpha=0.7, ax=ax)
ax.set_title("Class Distribution (Before Resampling)")
ax.set_xlabel("Potability (0: Not Potable, 1: Potable)")
ax.set_ylabel("Count")
st.pyplot(fig)

# Resample to balance classes
class_0 = data[data['Potability'] == 0]
class_1 = data[data['Potability'] == 1]
class_1_resampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
data_resampled = pd.concat([class_0, class_1_resampled])

# Visualize resampled distribution
st.subheader("Resampled Class Distribution")
fig, ax = plt.subplots()
data_resampled['Potability'].value_counts().plot(kind='bar', color=['blue', 'orange'], alpha=0.7, ax=ax)
ax.set_title("Class Distribution (After Resampling)")
ax.set_xlabel("Potability (0: Not Potable, 1: Potable)")
ax.set_ylabel("Count")
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap")
st.pyplot(fig)

# Histogram of features
st.subheader("Feature Distributions")
fig, ax = plt.subplots(figsize=(15, 10))
data.hist(bins=20, color='skyblue', edgecolor='black', ax=ax)
plt.tight_layout()
st.pyplot(fig)

# Split data and scale
X = data_imputed.drop('Potability', axis=1)
y = data_imputed['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and evaluation
st.subheader("Model Evaluation")
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

    st.write(f"**{model_name}**")
    st.write("Accuracy:", round(acc, 2))
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
                xticklabels=["Not Potable", "Potable"], 
                yticklabels=["Not Potable", "Potable"], ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}")
    st.pyplot(fig)

# Display accuracy comparison
st.subheader("Accuracy Comparison")
st.write(results)
