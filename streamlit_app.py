import streamlit as st
import pandas as pd
import joblib

# =========================
# Load the trained model
# =========================
@st.cache_data
def load_model():
    model_data = joblib.load("optimized_lvedp_model_features.joblib")
    return model_data

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

# =========================
# Streamlit App
# =========================
st.title("LVEDP Prediction App")
st.write("Enter patient data to predict LVEDP:")

# ------------------------
# User Inputs
# ------------------------
input_data = {}
for feat in features:
    # For simplicity, all features are number inputs
    input_data[feat] = st.number_input(feat, value=0.0)

# ------------------------
# Prediction Button
# ------------------------
if st.button("Predict LVEDP"):
    X_input = pd.DataFrame([input_data])
    # Scale features
    X_scaled = scaler.transform(X_input)
    # Predict
    pred = model.predict(X_scaled)
    st.success(f"Predicted LVEDP: {pred[0]:.2f} mmHg")
