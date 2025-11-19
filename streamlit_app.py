import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
@st.cache_data
def load_model():
    model_path = "optimized_lvedp_model_cv.joblib"
    model_data = joblib.load(model_path)
    return model_data

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']

# فقط الـ 8 Features اللي عايزاها
features = ['LVEF (%)', 'Ischemia duration (hr) ', 'LV global long. Strain (%)',
            'LA reservoir (%)', 'E/E`', 'Non-Ant STEMI', 'LA contraction (%)', 'Mitral E velocity (cm/s)']

st.title("LVEDP Prediction App")
st.write("Choose prediction type:")

prediction_type = st.radio("Prediction Mode", ("Single Prediction", "Batch Prediction"))

# ------------------------
# Single Prediction
# ------------------------
if prediction_type == "Single Prediction":
    st.subheader("Enter Patient Data")
    
    input_data = {}
    input_data['LVEF (%)'] = st.slider("LVEF (%)", 10, 80, 55)
    input_data['Ischemia duration (hr) '] = st.slider("Ischemia duration (hr)", 0, 24, 3)
    input_data['LV global long. Strain (%)'] = st.slider("LV global long. Strain (%)", -30, 0, -15)
    input_data['LA reservoir (%)'] = st.slider("LA reservoir (%)", 0, 80, 40)
    input_data['E/E`'] = st.slider("E/E`", 0, 30, 10)
    input_data['Non-Ant STEMI'] = st.selectbox("Non-Ant STEMI", [0, 1])
    input_data['LA contraction (%)'] = st.slider("LA contraction (%)", 0, 50, 25)
    input_data['Mitral E velocity (cm/s)'] = st.slider("Mitral E velocity (cm/s)", 20, 150, 80)
    
    if st.button("Predict LVEDP"):
        X_input = pd.DataFrame([input_data])
        X_scaled = scaler.transform(X_input[features])
        pred = model.predict(X_scaled)
        st.success(f"Predicted LVEDP: {pred[0]:.2f} mmHg")

# ------------------------
# Batch Prediction
# ------------------------
else:
    st.subheader("Upload Excel for Batch Prediction")
    uploaded_file = st.file_uploader("Choose Excel file", type=["xlsx", "xls"])
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"Missing columns in Excel: {missing_features}")
        else:
            X_batch = df[features]
            X_scaled = scaler.transform(X_batch)
            predictions = model.predict(X_scaled)
            
            df['Predicted LVEDP'] = predictions
            st.success("Predictions Completed!")
            st.dataframe(df)
            
            # Download results
            output_file = "LVEDP_Predictions.xlsx"
            df.to_excel(output_file, index=False)
            st.download_button("Download Predictions", output_file)
