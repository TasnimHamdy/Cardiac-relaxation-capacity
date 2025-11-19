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
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div style='text-align: center; background-color: #E8F0F2; padding: 15px; border-radius: 10px'>
        <h1 style='color: #2C3E50;'>LVEDP Prediction App</h1>
        <p style='color: #34495E;'>Enter patient data to predict LVEDP (mmHg)</p>
    </div>
    """, unsafe_allow_html=True
)

st.subheader("Patient Data Input")
cols = st.columns(2)  # تقسيم الأعمدة

input_data = {}
for i, feat in enumerate(features):
    with cols[i % 2]:
        # تحويل Non-Ant STEMI لرقم 0 أو 1
        if feat.lower() == "non-ant stemi":
            input_data[feat] = st.selectbox(
                feat, options=[0, 1], index=0, help="0 = No, 1 = Yes"
            )
        else:
            input_data[feat] = st.number_input(
                feat, value=0.0, step=0.1, format="%.2f"
            )

st.markdown("<hr>", unsafe_allow_html=True)

if st.button("Predict LVEDP"):
    X_input = pd.DataFrame([input_data])
    # Scale features
    X_scaled = scaler.transform(X_input)
    # Predict
    pred = model.predict(X_scaled)
    st.success(f"Predicted LVEDP: {pred[0]:.2f} mmHg")

