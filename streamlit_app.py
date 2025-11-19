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
    page_title="LVEDP Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div style='text-align: center; background-color: #f2f2f2; padding: 10px; border-radius: 10px'>
        <h1 style='color: #333333;'>LVEDP Prediction App</h1>
        <p>Enter patient data to predict LVEDP (mmHg)</p>
    </div>
    """, unsafe_allow_html=True
)

# ------------------------
# User Inputs in Columns
# ------------------------
st.subheader("Patient Data Input")
cols = st.columns(2)  # 2 columns for better layout

input_data = {}
for i, feat in enumerate(features):
    with cols[i % 2]:
        input_data[feat] = st.number_input(feat, value=0.0, step=0.1)

# ------------------------
# Prediction Button
# ------------------------
st.markdown("<hr>", unsafe_allow_html=True)
if st.button("Predict LVEDP"):
    X_input = pd.DataFrame([input_data])
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)
    st.success(f"Predicted LVEDP: {pred[0]:.2f} mmHg")
    st.balloons()  # Fun effect when prediction is done
