# =====================================
# STREAMLIT APP - LVEDP PREDICTION & VISUALIZATION
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="LVEDP Prediction & Analysis", layout="wide")

# ===========================
# 1. Load Model
# ===========================
@st.cache_resource
def load_model(path="optimized_lvedp_model_final.joblib"):
    model_data = joblib.load(path)
    return model_data

model_data = load_model()
optimized_ensemble = model_data["model"]
features = model_data["features"]
scaler = model_data["scaler"]
target = model_data["target"]

st.title("🏥 LVEDP Prediction & Statistical Analysis")
st.markdown("Upload your patient dataset (CSV or Excel) for prediction and analysis.")

# ===========================
# 2. File Uploader
# ===========================
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    st.success(f"Data loaded successfully! Shape: {df.shape}")
    
    # Ensure required features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.warning(f"Missing features in uploaded data: {missing_features}")
    
    # ===========================
    # 3. Predictions
    # ===========================
    df_features = df[features].copy()
    df_features = df_features.fillna(df_features.median())  # fill missing
    X_scaled = scaler.transform(df_features)
    predictions = optimized_ensemble.predict(X_scaled)
    df["Predicted_LVEDP"] = predictions
    
    st.subheader("Predictions")
    st.dataframe(df[[*features, "Predicted_LVEDP"]].head(20))
    
    # ===========================
    # 4. Statistical Visualizations
    # ===========================
    st.subheader("📊 Statistical Plots")
    
    # Feature distributions
    st.markdown("**Feature Distributions**")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,8))
    axes = axes.flatten()
    for i, col in enumerate(features[:6]):  # show first 6 features
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue', bins=15)
        axes[i].set_title(f'{col}')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Correlation heatmap
    st.markdown("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10,8))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    # Actual vs Predicted (if actual target exists)
    if target in df.columns:
        st.markdown("**Actual vs Predicted**")
        fig, ax = plt.subplots(figsize=(7,6))
        sns.scatterplot(x=df[target], y=df["Predicted_LVEDP"], s=70, color='blue', alpha=0.6, ax=ax)
        ax.plot([df[target].min(), df[target].max()], [df[target].min(), df[target].max()], 'r--', linewidth=2)
        ax.set_xlabel("Actual LVEDP")
        ax.set_ylabel("Predicted LVEDP")
        st.pyplot(fig)
        
        # Residuals
        st.markdown("**Residual Analysis**")
        residuals = df[target] - df["Predicted_LVEDP"]
        fig, ax = plt.subplots(figsize=(7,6))
        sns.scatterplot(x=df["Predicted_LVEDP"], y=residuals, s=60, color='purple', alpha=0.6, ax=ax)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        st.pyplot(fig)
        
        # Error Distribution
        st.markdown("**Absolute Error Distribution**")
        fig, ax = plt.subplots(figsize=(7,6))
        sns.histplot(np.abs(residuals), bins=20, kde=True, color='orange', ax=ax)
        ax.axvline(np.mean(np.abs(residuals)), color='red', linestyle='--', label=f"Mean Abs Error: {np.mean(np.abs(residuals)):.2f}")
        ax.set_xlabel("Absolute Error")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
    
    # Feature Importance (if RandomForest exists)
    st.markdown("**Feature Importance (Random Forest)**")
    rf_model = optimized_ensemble.estimators_[0]
    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
        st.pyplot(fig)

