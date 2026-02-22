import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------------------------
# App Title
# ---------------------------------
st.title("🏦 Bank Loan Prediction using Random Forest")

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bank.csv")
    return df

df = load_data()

st.subheader("📌 Dataset Preview")
st.write(df.head())

# ---------------------------------
# Preprocessing
# ---------------------------------
data = df.copy()

le = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])

# Features & Target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------
# Model Training
# ---------------------------------
st.subheader("🌲 Random Forest Model")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------------
# Evaluation
# ---------------------------------
st.subheader("📊 Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)
st.write("Accuracy:", round(accuracy * 100, 2), "%")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
st.pyplot(fig)

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------------------------------
# Prediction Section
# ---------------------------------
st.subheader("🔮 Make Prediction")

input_data = []

for col in X.columns:
    value = st.number_input(f"Enter {col}", value=0.0)
    input_data.append(value)

if st.button("Predict Loan Status"):
    input_array = np.array(input_data).reshape(1, -1)
    input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")