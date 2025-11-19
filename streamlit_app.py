# streamlit_app.py
from joblib import load
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import io
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="LVEDP Prediction Tool",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .feature-input {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model"""
    try:
        return load("optimized_lvedp_model.joblib")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please make sure 'optimized_lvedp_model.joblib' is in the same directory.")
        return None
    except FileNotFoundError:
        st.error("❌ Model file not found. Please make sure 'optimized_lvedp_model.pkl' is in the same directory.")
        st.info("💡 Run 'train_model.py' first to train and save the model.")
        return None

def predict_lvedp(model_data, input_features):
    """Predict LVEDP for given features"""
    try:
        # Scale the input features
        input_scaled = model_data['scaler'].transform([input_features])
        
        # Make prediction
        prediction = model_data['model'].predict(input_scaled)[0]
        
        # Calculate confidence interval (based on test MAE)
        confidence = 1.96 * model_data['performance']['test_mae']
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ LVEDP Prediction Tool</h1>', unsafe_allow_html=True)
    st.write("Predict Left Ventricular End-Diastolic Pressure using clinical parameters")
    
    # Load model
    model_data = load_model()
    if model_data is None:
        return
    
    # Sidebar for navigation
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

def single_prediction_mode(model_data):
    """Single prediction interface"""
    st.header("🔍 Single Patient Prediction")
    
    # Create two columns for feature inputs
    col1, col2 = st.columns(2)
    
    input_features = []
    features = model_data['features']
    
    with col1:
        st.subheader("Clinical Parameters")
        for i, feature in enumerate(features[:len(features)//2]):
            # Provide reasonable default ranges based on typical clinical values
            if "LVEF" in feature:
                value = st.number_input(f"{feature}", min_value=10.0, max_value=80.0, value=55.0, step=1.0)
            elif "E/E" in feature:
                value = st.number_input(f"{feature}", min_value=5.0, max_value=25.0, value=10.0, step=0.5)
            elif "duration" in feature.lower():
                value = st.number_input(f"{feature}", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
            elif "reservoir" in feature.lower():
                value = st.number_input(f"{feature}", min_value=10.0, max_value=60.0, value=35.0, step=1.0)
            elif "STEMI" in feature:
                value = st.selectbox(f"{feature}", [0, 1])
            elif "Group" in feature:
                value = st.number_input(f"{feature}", min_value=1.0, max_value=3.0, value=2.0, step=1.0)
            else:
                value = st.number_input(f"{feature}", value=0.0, step=0.1)
            
            input_features.append(value)
    
    with col2:
        st.subheader("Additional Parameters")
        for i, feature in enumerate(features[len(features)//2:]):
            if "LVEF" in feature:
                value = st.number_input(f"{feature}", min_value=10.0, max_value=80.0, value=55.0, step=1.0)
            elif "E/E" in feature:
                value = st.number_input(f"{feature}", min_value=5.0, max_value=25.0, value=10.0, step=0.5)
            elif "duration" in feature.lower():
                value = st.number_input(f"{feature}", min_value=0.0, max_value=24.0, value=6.0, step=0.5)
            elif "reservoir" in feature.lower():
                value = st.number_input(f"{feature}", min_value=10.0, max_value=60.0, value=35.0, step=1.0)
            elif "STEMI" in feature:
                value = st.selectbox(f"{feature}", [0, 1])
            elif "Group" in feature:
                value = st.number_input(f"{feature}", min_value=1.0, max_value=3.0, value=2.0, step=1.0)
            else:
                value = st.number_input(f"{feature}", value=0.0, step=0.1)
            
            input_features.append(value)
    
    # Prediction button
    if st.button("🎯 Predict LVEDP", type="primary", use_container_width=True):
        if len(input_features) == len(features):
            with st.spinner("Calculating prediction..."):
                prediction, confidence = predict_lvedp(model_data, input_features)
                
                if prediction is not None:
                    # Display results
                    st.markdown("---")
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Predicted LVEDP",
                            value=f"{prediction:.1f} mmHg",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence Range",
                            value=f"±{confidence:.1f} mmHg"
                        )
                    
                    with col3:
                        # Clinical interpretation
                        if prediction < 16:
                            status = "📗 Normal"
                            interpretation = "LVEDP within normal range"
                            color = "green"
                        elif prediction < 20:
                            status = "📙 Borderline"
                            interpretation = "Monitor closely"
                            color = "orange"
                        else:
                            status = "📕 Elevated"
                            interpretation = "Clinical attention needed"
                            color = "red"
                        
                        st.metric(
                            label="Clinical Status",
                            value=status
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Interpretation
                    st.info(f"**Clinical Interpretation:** {interpretation}")
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 2))
                    categories = ['Normal (<16)', 'Borderline (16-20)', 'Elevated (>20)']
                    ranges = [(10, 16), (16, 20), (20, 30)]
                    colors = ['green', 'orange', 'red']
                    
                    for i, (start, end) in enumerate(ranges):
                        ax.barh(0, end-start, left=start, color=colors[i], alpha=0.3, height=0.5)
                    
                    ax.axvline(prediction, color='blue', linewidth=3, label=f'Prediction: {prediction:.1f} mmHg')
                    ax.set_xlim(10, 30)
                    ax.set_yticks([])
                    ax.set_xlabel('LVEDP (mmHg)')
                    ax.set_title('Prediction in Clinical Context')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                else:
                    st.error("❌ Prediction failed. Please check your inputs.")
        else:
            st.error("❌ Please fill all the input fields.")

def batch_prediction_mode(model_data):
    """Batch prediction interface"""
    st.header("📊 Batch Prediction")
    
    st.write("Upload an Excel file with patient data for batch predictions")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose Excel file", 
        type=['xlsx', 'xls'],
        help="Upload Excel file with the same features as the model"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded successfully! Found {len(df)} patients")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Check if required features are present
            missing_features = set(model_data['features']) - set(df.columns)
            if missing_features:
                st.error(f"❌ Missing features in uploaded file: {missing_features}")
                return
            
            # Make predictions
            if st.button("🚀 Run Batch Predictions", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    # Select only the required features
                    X_batch = df[model_data['features']]
                    
                    # Scale features
                    X_scaled = model_data['scaler'].transform(X_batch)
                    
                    # Make predictions
                    predictions = model_data['model'].predict(X_scaled)
                    
                    # Add predictions to dataframe
                    results_df = df.copy()
                    results_df['Predicted_LVEDP'] = predictions
                    results_df['Confidence_Range'] = 1.96 * model_data['performance']['test_mae']
                    
                    # Add clinical interpretation
                    def interpret_lvedp(lvedp):
                        if lvedp < 16:
                            return 'Normal'
                        elif lvedp < 20:
                            return 'Borderline'
                        else:
                            return 'Elevated'
                    
                    results_df['Clinical_Status'] = results_df['Predicted_LVEDP'].apply(interpret_lvedp)
                    
                    # Display results
                    st.subheader("📈 Prediction Results")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="lvedp_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Patients", len(results_df))
                    with col2:
                        normal_count = len(results_df[results_df['Clinical_Status'] == 'Normal'])
                        st.metric("Normal", normal_count)
                    with col3:
                        borderline_count = len(results_df[results_df['Clinical_Status'] == 'Borderline'])
                        st.metric("Borderline", borderline_count)
                    with col4:
                        elevated_count = len(results_df[results_df['Clinical_Status'] == 'Elevated'])
                        st.metric("Elevated", elevated_count)
                    
                    # Distribution plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    status_counts = results_df['Clinical_Status'].value_counts()
                    colors = ['green', 'orange', 'red']
                    bars = ax.bar(status_counts.index, status_counts.values, color=colors, alpha=0.7)
                    
                    ax.set_ylabel('Number of Patients')
                    ax.set_title('Distribution of Clinical Status')
                    ax.grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar, count in zip(bars, status_counts.values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{count}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

def model_information_mode(model_data):
    """Model information and performance"""
    st.header("ℹ️ Model Information")
    
    # Model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Specifications")
        st.write(f"**Model Type:** {model_data['model_type']}")
        st.write(f"**Target Variable:** {model_data['target']}")
        st.write(f"**Number of Features:** {len(model_data['features'])}")
        st.write(f"**Training Samples:** {model_data['data_info']['training_samples']}")
        st.write(f"**Test Samples:** {model_data['data_info']['test_samples']}")
    
    with col2:
        st.subheader("Performance Metrics")
        perf = model_data['performance']
        st.write(f"**Test MAE:** {perf['test_mae']:.3f} mmHg")
        st.write(f"**Test R²:** {perf['test_r2']:.3f}")
        st.write(f"**Overfitting Gap:** {perf['overfitting_gap']:.3f}")
        st.write(f"**Cross-validation MAE:** {perf['cv_mae']:.3f} ± {perf['cv_std']:.3f}")
    
    # Features list
    st.subheader("Selected Features")
    features_df = pd.DataFrame({
        'Feature': model_data['features'],
        'Description': 'Clinical parameter used for prediction'
    })
    st.dataframe(features_df, use_container_width=True)
    
    # Performance visualization
    st.subheader("Performance Visualization")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. MAE comparison
    models = ['Train', 'Test']
    mae_scores = [perf['train_mae'], perf['test_mae']]
    colors = ['blue', 'orange']
    
    bars = ax1.bar(models, mae_scores, color=colors, alpha=0.7)
    ax1.set_ylabel('MAE (mmHg)')
    ax1.set_title('Model Performance: Train vs Test')
    ax1.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, mae_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 2. Overfitting gap
    ax2.bar(['Overfitting Gap'], [perf['overfitting_gap']], 
            color='red' if perf['overfitting_gap'] > 0.3 else 'green', alpha=0.7)
    ax2.set_ylabel('MAE Difference')
    ax2.set_title('Overfitting Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.text(0, perf['overfitting_gap'] + 0.01, f'{perf["overfitting_gap"]:.3f}', 
             ha='center', va='bottom')
    
    # 3. R² score
    ax3.bar(['R² Score'], [perf['test_r2']], color='purple', alpha=0.7)
    ax3.set_ylabel('R² Value')
    ax3.set_title('Model Explanation Power')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.text(0, perf['test_r2'] + 0.02, f'{perf["test_r2"]:.3f}', 
             ha='center', va='bottom')
    
    # 4. Feature importance (if available)
    try:
        model = model_data['model'].estimators_[0]  # Get first base model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': model_data['features'],
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            ax4.barh(feature_imp_df['feature'], feature_imp_df['importance'], color='lightgreen')
            ax4.set_xlabel('Importance Score')
            ax4.set_title('Feature Importance (Random Forest)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance')
    except:
        ax4.text(0.5, 0.5, 'Feature Importance\nNot Available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Feature Importance')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Clinical guidelines
    st.subheader("📋 Clinical Interpretation Guidelines")
    
    guidelines = """
    | LVEDP Range (mmHg) | Clinical Status | Interpretation |
    |-------------------|-----------------|----------------|
    | < 16 | Normal | Within normal physiological range |
    | 16 - 20 | Borderline | Requires monitoring and follow-up |
    | > 20 | Elevated | Clinical attention needed, consider intervention |
    
    **Note:** These predictions are for decision support and should be interpreted by qualified healthcare professionals.
    """
    
    st.markdown(guidelines)

if __name__ == "__main__":
    main()