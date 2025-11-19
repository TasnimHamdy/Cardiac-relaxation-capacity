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
st.set_page_config(
    page_title="LVEDP Prediction",
    page_icon="❤️",
    layout="centered"
)

st.markdown("<h1 style='text-align:center; color:red;'>❤️ LVEDP Prediction App</h1>", unsafe_allow_html=True)
st.markdown("### Enter patient data to predict LVEDP (mmHg)")

# ------------------------
# User Inputs in 2 columns
# ------------------------
input_data = {}
cols = st.columns(2)
for i, feat in enumerate(features):
    col = cols[i % 2]
    if feat == "Ant STEMI":
        input_data[feat] = col.selectbox(
            feat, options=[0, 1], index=0, help="0 = No, 1 = Yes"
        )
    else:
        input_data[feat] = col.number_input(
            feat, value=0.0, step=0.1, format="%.2f"
        )

st.markdown("---")

# ------------------------
# Prediction Button
# ------------------------
if st.button("Predict LVEDP ❤️"):
    X_input = pd.DataFrame([input_data])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)
    st.success(f"Predicted LVEDP: {pred[0]:.2f} mmHg")
