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
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("❤️ LVEDP Prediction App")
st.write("Enter patient data to predict LVEDP (mmHg)")

# ------------------------
# User Inputs
# ------------------------
input_data = {}
for feat in features:
    if feat == "Ant STEMI":
        input_data[feat] = st.selectbox(
            feat, options=[0, 1], index=0, help="0 = No, 1 = Yes"
        )
    else:
        input_data[feat] = st.number_input(
            feat, value=0.0, step=0.1, format="%.2f"
        )

st.markdown("---")

# ------------------------
# Prediction Button
# ------------------------
if st.button("Predict LVEDP"):
    X_input = pd.DataFrame([input_data])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)
    st.success(f"Predicted LVEDP: {pred[0]:.2f} mmHg")


