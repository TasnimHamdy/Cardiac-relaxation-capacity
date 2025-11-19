import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# 1) LOAD MODEL
# ==========================
@st.cache_data
def load_model():
    model_path = "optimized_lvedp_model_cv.joblib"
    model_data = joblib.load(model_path)
    return model_data

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']

# Define features explicitly
features = [
    "LVEF (%)",
    "Ischemia duration (hr)",
    "LV global long. Strain (%)",
    "LA reservoir (%)",
    "E/E`",
    "Non-Ant STEMI",
    "LA contraction (%)",
    "Mitral E velocity (cm/s)"
]

st.title("LVEDP Prediction App")
st.write("### Choose Prediction Mode")

prediction_type = st.radio("Prediction Mode", ("Single Prediction", "Batch Prediction"))

# ==========================
# 2) SINGLE PREDICTION
# ==========================
if prediction_type == "Single Prediction":
    st.subheader("Enter Patient Data Manually")
    input_data = {}
    for feat in features:
        input_data[feat] = st.number_input(feat, value=0.0)
    
    if st.button("Predict LVEDP"):
        X_input = pd.DataFrame([input_data])
        X_scaled = scaler.transform(X_input)
        pred = model.predict(X_scaled)
        st.success(f"Predicted LVEDP: {pred[0]:.2f} mmHg")

# ==========================
# 3) BATCH PREDICTION
# ==========================
else:
    st.subheader("Upload Excel File for Batch Prediction")
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
            st.success("✅ Predictions Completed!")
            st.dataframe(df)
            
            # Allow user to download results
            output_file = "LVEDP_Predictions.xlsx"
            df.to_excel(output_file, index=False)
            st.download_button("Download Predictions", output_file)

