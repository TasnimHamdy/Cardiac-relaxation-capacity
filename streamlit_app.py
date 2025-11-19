# streamlit_app.py
from joblib import load
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="LVEDP Prediction Tool",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
def load_model():
    """Load the trained joblib model safely"""
    model_path = "optimized_lvedp_model.joblib"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at {model_path}")
        st.info("💡 Please upload 'optimized_lvedp_model.joblib' to this directory.")
        return None
    try:
        model_data = load(model_path)
        return model_data
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# ------------------ Prediction Function ------------------
def predict_lvedp(model_data, input_features):
    """Predict LVEDP for given features"""
    try:
        input_scaled = model_data['scaler'].transform([input_features])
        prediction = model_data['model'].predict(input_scaled)[0]
        confidence = 1.96 * model_data['performance']['test_mae']
        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ------------------ Main App ------------------
def main():
    st.markdown('<h1 class="main-header">❤️ LVEDP Prediction Tool</h1>', unsafe_allow_html=True)
    st.write("Predict Left Ventricular End-Diastolic Pressure using clinical parameters")
    
    # Load model
    model_data = load_model()
    if model_data is None:
        return
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Single Prediction", "Batch Prediction", "Model Information"]
    )
    
    if app_mode == "Single Prediction":
        single_prediction_mode(model_data)
    elif app_mode == "Batch Prediction":
        batch_prediction_mode(model_data)
    else:
        model_information_mode(model_data)

# ------------------ Single Prediction ------------------
def single_prediction_mode(model_data):
    st.header("🔍 Single Patient Prediction")
    features = model_data['features']
    input_features = []

    col1, col2 = st.columns(2)
    for col, feats in zip([col1, col2], [features[:len(features)//2], features[len(features)//2:]]):
        with col:
            for feature in feats:
                if "LVEF" in feature:
                    value = st.number_input(feature, min_value=10.0, max_value=80.0, value=55.0)
                elif "E/E" in feature:
                    value = st.number_input(feature, min_value=5.0, max_value=25.0, value=10.0)
                elif "duration" in feature.lower():
                    value = st.number_input(feature, min_value=0.0, max_value=24.0, value=6.0)
                elif "reservoir" in feature.lower():
                    value = st.number_input(feature, min_value=10.0, max_value=60.0, value=35.0)
                elif "STEMI" in feature:
                    value = st.selectbox(feature, [0,1])
                elif "Group" in feature:
                    value = st.number_input(feature, min_value=1, max_value=3, value=2)
                else:
                    value = st.number_input(feature, value=0.0, step=0.1)
                input_features.append(value)
    
    if st.button("🎯 Predict LVEDP"):
        if len(input_features) == len(features):
            prediction, confidence = predict_lvedp(model_data, input_features)
            if prediction is not None:
                st.markdown(f"### Predicted LVEDP: {prediction:.1f} mmHg ± {confidence:.1f}")
                if prediction < 16:
                    st.success("📗 Normal")
                elif prediction < 20:
                    st.warning("📙 Borderline")
                else:
                    st.error("📕 Elevated")
        else:
            st.error("❌ Please fill all input fields.")

# ------------------ Batch Prediction ------------------
def batch_prediction_mode(model_data):
    st.header("📊 Batch Prediction")
    
    st.write("Upload an Excel file with patient data (features should match model features).")
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx','xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ File loaded: {len(df)} patients")
            st.dataframe(df.head())
            
            missing = set(model_data['features']) - set(df.columns)
            if missing:
                st.error(f"Missing features in uploaded file: {missing}")
                return
            
            if st.button("🚀 Run Batch Predictions"):
                X = df[model_data['features']]
                X_scaled = model_data['scaler'].transform(X)
                preds = model_data['model'].predict(X_scaled)
                df['Predicted_LVEDP'] = preds
                df['Confidence'] = 1.96 * model_data['performance']['test_mae']
                
                def interpret(x):
                    if x < 16: return 'Normal'
                    elif x < 20: return 'Borderline'
                    else: return 'Elevated'
                df['Clinical_Status'] = df['Predicted_LVEDP'].apply(interpret)
                
                st.subheader("📈 Prediction Results")
                st.dataframe(df)
                
                csv = df.to_csv(index=False)
                st.download_button("📥 Download CSV", data=csv, file_name="lvedp_predictions.csv")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# ------------------ Model Info ------------------
def model_information_mode(model_data):
    st.header("ℹ️ Model Information")
    st.write(f"**Model Type:** {model_data['model_type']}")
    st.write(f"**Target:** {model_data['target']}")
    st.write(f"Number of Features: {len(model_data['features'])}")
    st.write(f"Training samples: {model_data['data_info']['training_samples']}")
    st.write(f"Test samples: {model_data['data_info']['test_samples']}")
    
    perf = model_data['performance']
    st.write(f"Test MAE: {perf['test_mae']:.3f}")
    st.write(f"Test R²: {perf['test_r2']:.3f}")

# ------------------ Run App ------------------
if __name__ == "__main__":
    main()
