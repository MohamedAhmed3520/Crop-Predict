import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv("crop.csv")
X = data.drop(["Unnamed: 0", "label"], axis=1)
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_tr, x_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
x_val = scaler.transform(x_val)

# Load trained model (must be saved using same scikit-learn version!)
model = joblib.load('crop.pkl')

# Streamlit UI
st.title("🌾 Crop Type Prediction")
st.subheader("Enter the required soil and climate conditions:")

N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Level")
rainfall = st.number_input("Rainfall (mm)")

label_decoder = {
    0: "apple",
    1: "banana",
    2: "blackgram",
    3: "chickpea",
    4: "coconut",
    5: "coffee",
    6: "cotton",
    7: "grapes",
    8: "jute",
    9: "kidneybeans",
    10: "lentil",
    11: "maize",
    12: "mango",
    13: "mothbeans",
    14: "mungbean",
    15: "muskmelon",
    16: "orange",
    17: "papaya",
    18: "pigeonpeas",
    19: "pomegranate",
    20: "rice",
    21: "watermelon"
}

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=X.columns)
        input_scaled = scaler.transform(input_df)
        predicted_label = model.predict(input_scaled)[0]
        crop_name = label_decoder.get(predicted_label, "Unknown")

        st.success(f"The predicted crop is: **{crop_name}**")
        st.balloons()

        st.write("✅ Model Validation Accuracy:")
        st.write(f"**{model.score(x_val, y_val) * 100:.2f}%**")

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction: {e}")
