import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Load Model
# -----------------------
@st.cache_data
def load_model():
    model_path = "optimized_lvedp_model_cv.joblib"
    model_data = joblib.load(model_path)
    return model_data

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']

# -----------------------
# Fixed Features List
# -----------------------
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
st.write("Select prediction mode:")

prediction_type = st.radio("Prediction Mode", ("Single Prediction", "Batch Prediction"))

# ------------------------
# Single Prediction
# ------------------------
if prediction_type == "Single Prediction":
    st.subheader("Enter Patient Data")
    
    input_data = {}
    for feat in features:
        if feat == "Non-Ant STEMI":
            input_data[feat] = st.selectbox(feat, options=[0,1])
        else:
            input_data[feat] = st.number_input(feat, value=0.0)
    
    if st.button("Predict LVEDP"):
        X_input = pd.DataFrame([input_data], columns=features)  # maintain order
        X_scaled = scaler.transform(X_input)
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
        df.columns = df.columns.str.strip()  # remove extra spaces
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(f"Missing columns in Excel: {missing_features}")
        else:
            X_batch = df[features]  # preserve order
            X_scaled = scaler.transform(X_batch)
            predictions = model.predict(X_scaled)
            df['Predicted LVEDP'] = predictions
            st.success("Batch Predictions Completed!")
            st.dataframe(df)
            
            # Download results
            output_file = "LVEDP_Predictions.xlsx"
            df.to_excel(output_file, index=False)
            st.download_button("Download Predictions", output_file)

