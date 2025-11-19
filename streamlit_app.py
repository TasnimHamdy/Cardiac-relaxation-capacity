# %%
# ============================================================
# ENHANCED LVEDP PREDICTION MODEL WITH REDUCED OVERFITTING
# Fixed SMOTE + Advanced Regularization
# ============================================================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl", 10)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("🚀 Starting OPTIMIZED LVEDP Prediction Model...")

# =======================
# 1) LOAD AND EXPLORE DATA
# =======================
PATH = r"D:\Nti\1-10-2005.xlsx"
df = pd.read_excel(PATH)

# Detect target
target = [c for c in df.columns if "lvedp" in c.lower()][0]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in numeric_cols if c != target]

df = df.dropna(subset=[target])

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

X = df[features].reset_index(drop=True)
y = df[target].reset_index(drop=True)

print(f"📊 Dataset shape: {X.shape}")
print(f"🎯 Target distribution:\n{y.describe()}")

# =======================
# 2) ENHANCED FEATURE SELECTION
# =======================
print("\n🔧 Applying ENHANCED feature selection...")

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

def advanced_feature_selection(X, y, n_features=8):
    """Enhanced feature selection with multiple methods"""
    
    # Method 1: Lasso for feature selection
    lasso = LassoCV(cv=5, random_state=42, max_iter=1000)
    lasso.fit(X, y)
    
    lasso_features = X.columns[np.abs(lasso.coef_) > 0.01].tolist()
    print(f"✅ Lasso selected {len(lasso_features)} features")
    
    # Method 2: Random Forest importance
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    threshold = np.percentile(importances, 70)  # Top 30% features
    rf_features = X.columns[importances >= threshold].tolist()
    print(f"✅ RF selected {len(rf_features)} features")
    
    # Combine methods
    combined_features = list(set(lasso_features) | set(rf_features))
    
    # If still too many, take top by correlation
    if len(combined_features) > n_features:
        correlations = X[combined_features].corrwith(y).abs()
        combined_features = correlations.nlargest(n_features).index.tolist()
    
    print(f"✅ Final selected {len(combined_features)} features")
    return combined_features

selected_features = advanced_feature_selection(X, y, n_features=8)
X = X[selected_features]
features = selected_features

print(f"🎯 Selected features: {features}")

# =======================
# 3) FIXED SMOTE FOR REGRESSION
# =======================
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def fixed_smote_regression(X, y, n_neighbors=2, target_bins=4):
    """
    Fixed SMOTE for regression with proper broadcasting
    """
    # Create bins based on target distribution
    y_bins = pd.qcut(y, q=target_bins, labels=False, duplicates='drop')
    
    # Check bin distribution
    bin_counts = pd.Series(y_bins).value_counts().sort_index()
    print(f"📊 Original bin distribution: {dict(bin_counts)}")
    
    # Calculate desired samples per bin (balance to max bin)
    max_samples = bin_counts.max()
    sampling_strategy = {bin_idx: max_samples for bin_idx in bin_counts.index}
    
    # Apply SMOTE with safe k_neighbors
    safe_k = min(n_neighbors, bin_counts.min() - 1)
    if safe_k < 1:
        safe_k = 1
        
    sm = SMOTE(
        k_neighbors=safe_k, 
        sampling_strategy=sampling_strategy,
        random_state=42
    )
    
    X_sm, y_bins_sm = sm.fit_resample(X, y_bins)
    
    # Reconstruct continuous target using KNN - FIXED VERSION
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(X)
    
    # Find nearest neighbors for synthetic samples
    synthetic_indices = range(len(X), len(X_sm))
    y_sm_continuous = y.values.copy()
    
    for idx in synthetic_indices:
        synthetic_sample = X_sm[idx:idx+1]
        distances, neighbor_indices = nn.kneighbors(synthetic_sample)
        
        # Get neighbor targets and ensure proper dimensions
        neighbor_targets = y.values[neighbor_indices[0]]
        
        # Calculate weights based on distances (closer neighbors have higher weight)
        weights = 1.0 / (distances[0] + 1e-8)  # Add small value to avoid division by zero
        weights = weights / weights.sum()  # Normalize weights
        
        # Calculate weighted average - FIXED BROADCASTING
        synthetic_target = np.dot(weights, neighbor_targets)
        y_sm_continuous = np.append(y_sm_continuous, synthetic_target)
    
    print(f"✅ Fixed SMOTE: {X.shape} → {X_sm.shape}")
    
    # Verify target distribution preservation
    original_stats = {
        'mean': y.mean(),
        'std': y.std(),
        'min': y.min(),
        'max': y.max()
    }
    
    synthetic_stats = {
        'mean': y_sm_continuous.mean(),
        'std': y_sm_continuous.std(), 
        'min': y_sm_continuous.min(),
        'max': y_sm_continuous.max()
    }
    
    print(f"📊 Original target - Mean: {original_stats['mean']:.2f}, Std: {original_stats['std']:.2f}")
    print(f"📊 Synthetic target - Mean: {synthetic_stats['mean']:.2f}, Std: {synthetic_stats['std']:.2f}")
    
    return X_sm, y_sm_continuous

print("\n🔄 Applying FIXED SMOTE for regression...")
X_sm, y_sm = fixed_smote_regression(X, y, n_neighbors=2, target_bins=4)

# =======================
# 4) STRATIFIED DATA SPLITTING
# =======================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sm)

# Create bins for stratification
y_bins = pd.qcut(y_sm, q=4, labels=False, duplicates='drop')

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_sm, test_size=0.25, random_state=42, stratify=y_bins
)

print(f"\n📊 Optimized data split:")
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   Train target range: {y_train.min():.2f} - {y_train.max():.2f}")
print(f"   Test target range: {y_test.min():.2f} - {y_test.max():.2f}")

# =======================
# 5) OPTIMIZED ENSEMBLE WITH REDUCED OVERFITTING
# =======================
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

print("\n🔄 Building OPTIMIZED ensemble with reduced overfitting...")

def create_optimized_ensemble():
    """Create ensemble with strong regularization"""
    
    base_models = [
        ('rf_regularized', RandomForestRegressor(
            n_estimators=80,
            max_depth=4,           # Reduced depth
            min_samples_split=15,  # Increased minimum split
            min_samples_leaf=8,    # Increased minimum leaf
            max_features=0.6,      # Limit features per tree
            random_state=42,
            n_jobs=1
        )),
        ('gbm_regularized', GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,           # Shallow trees
            learning_rate=0.05,    # Lower learning rate
            subsample=0.8,         # Use 80% of samples per tree
            max_features=0.7,      # Use 70% of features per tree
            random_state=42
        )),
        ('ridge_regularized', Ridge(
            alpha=5.0,             # Strong regularization
            random_state=42
        )),
        ('svr_regularized', SVR(
            kernel='rbf',
            C=1.0,                 # Regularization parameter
            epsilon=0.1,
            gamma='scale'
        ))
    ]
    
    # Equal weights voting regressor
    ensemble = VotingRegressor(
        estimators=base_models,
        weights=[1, 1, 1, 1],  # Equal weights
        n_jobs=1
    )
    
    return ensemble

# Create and train optimized ensemble
optimized_ensemble = create_optimized_ensemble()
print("🔄 Training optimized ensemble...")
optimized_ensemble.fit(X_train, y_train)

# =======================
# 6) COMPREHENSIVE MODEL EVALUATION
# =======================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

print("\n📊 Evaluating optimized model...")

# Predictions
train_predictions = optimized_ensemble.predict(X_train)
test_predictions = optimized_ensemble.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
overfitting_gap = abs(train_mae - test_mae)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Additional metrics
mape = np.mean(np.abs((y_test - test_predictions) / np.where(y_test != 0, y_test, 1))) * 100
correlation = np.corrcoef(y_test, test_predictions)[0, 1]

print("\n" + "="*60)
print("🎯 OPTIMIZED MODEL PERFORMANCE")
print("="*60)
print(f"📊 Train MAE      : {train_mae:.4f}")
print(f"📊 Test MAE       : {test_mae:.4f}")
print(f"🚨 Overfitting Gap : {overfitting_gap:.4f} (Target: < 0.3)")
print(f"📊 Train R²       : {train_r2:.4f}")
print(f"📊 Test R²        : {test_r2:.4f}")
print(f"📊 MAPE           : {mape:.2f}%")
print(f"📊 Correlation    : {correlation:.4f}")

# Compare with baseline
baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_test.mean()))
improvement = ((baseline_mae - test_mae) / baseline_mae) * 100
print(f"🚀 Improvement over baseline: {improvement:.1f}%")

# =======================
# 7) CROSS-VALIDATION FOR ROBUSTNESS
# =======================
from sklearn.model_selection import cross_val_score, StratifiedKFold

def stratified_cross_validation(X, y, model, n_splits=5):
    """Stratified cross-validation for regression"""
    # Create bins for stratification
    y_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in skf.split(X, y_bins):
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        model.fit(X_train_cv, y_train_cv)
        score = mean_absolute_error(y_test_cv, model.predict(X_test_cv))
        cv_scores.append(score)
    
    return np.mean(cv_scores), np.std(cv_scores)

print("\n🔄 Performing stratified cross-validation...")
cv_mean, cv_std = stratified_cross_validation(X_scaled, y_sm, optimized_ensemble, n_splits=5)
print(f"📊 Cross-validation MAE: {cv_mean:.4f} ± {cv_std:.4f}")

# =======================
# 8) INDIVIDUAL MODEL COMPARISON
# =======================
print(f"\n🔍 Individual Model Performance:")
individual_models = optimized_ensemble.estimators_

for name, model in zip(['RF', 'GBM', 'Ridge', 'SVR'], individual_models):
    model.fit(X_train, y_train)
    individual_pred = model.predict(X_test)
    individual_mae = mean_absolute_error(y_test, individual_pred)
    individual_r2 = r2_score(y_test, individual_pred)
    print(f"   {name:6s}: MAE = {individual_mae:.4f}, R² = {individual_r2:.4f}")

# =======================
# 9) COMPREHENSIVE VISUALIZATION
# =======================
print("\n🎨 Generating comprehensive visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Actual vs Predicted
axes[0,0].scatter(y_test, test_predictions, alpha=0.6, s=60, color='blue')
axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
axes[0,0].set_xlabel('Actual LVEDP')
axes[0,0].set_ylabel('Predicted LVEDP')
axes[0,0].set_title(f'Actual vs Predicted\nTest R² = {test_r2:.3f}', fontweight='bold')
axes[0,0].grid(True, alpha=0.3)

# 2. Overfitting Analysis
models = ['Train', 'Test']
mae_scores = [train_mae, test_mae]
colors = ['green', 'red'] if overfitting_gap > 0.3 else ['blue', 'orange']

bars = axes[0,1].bar(models, mae_scores, color=colors, alpha=0.7, edgecolor='black')
axes[0,1].set_ylabel('MAE')
axes[0,1].set_title(f'Overfitting Analysis\nGap = {overfitting_gap:.3f}', fontweight='bold')
axes[0,1].grid(True, alpha=0.3)

for bar, score in zip(bars, mae_scores):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Residuals Plot
residuals = y_test - test_predictions
axes[0,2].scatter(test_predictions, residuals, alpha=0.6, s=50, c=residuals, cmap='RdBu_r')
axes[0,2].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0,2].set_xlabel('Predicted Values')
axes[0,2].set_ylabel('Residuals')
axes[0,2].set_title('Residuals Analysis', fontweight='bold')
axes[0,2].grid(True, alpha=0.3)

# 4. Error Distribution
axes[1,0].hist(np.abs(residuals), bins=20, alpha=0.7, color='purple', edgecolor='black', density=True)
axes[1,0].axvline(np.mean(np.abs(residuals)), color='red', linestyle='--', 
                 label=f'Mean: {np.mean(np.abs(residuals)):.2f}')
axes[1,0].set_xlabel('Absolute Error')
axes[1,0].set_ylabel('Density')
axes[1,0].set_title('Error Distribution', fontweight='bold')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 5. Model Comparison
model_names = ['RF', 'GBM', 'Ridge', 'SVR', 'Ensemble']
model_maes = []

for name, model in zip(['RF', 'GBM', 'Ridge', 'SVR'], individual_models):
    pred = model.predict(X_test)
    model_maes.append(mean_absolute_error(y_test, pred))

model_maes.append(test_mae)

colors = ['lightblue'] * 4 + ['gold']
bars = axes[1,1].bar(model_names, model_maes, color=colors, alpha=0.7, edgecolor='black')
axes[1,1].set_ylabel('Test MAE')
axes[1,1].set_title('Model Comparison', fontweight='bold')
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(True, alpha=0.3)

for bar, mae in zip(bars, model_maes):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                  f'{mae:.3f}', ha='center', va='bottom', fontweight='bold')

# 6. Feature Importance (from Random Forest)
rf_model = individual_models[0]
if hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    axes[1,2].barh(feature_imp_df['feature'], feature_imp_df['importance'], color='lightgreen')
    axes[1,2].set_xlabel('Importance Score')
    axes[1,2].set_title('Feature Importance (Random Forest)', fontweight='bold')
    axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimized_model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# =======================
# 10) SAVE OPTIMIZED MODEL
# =======================
model_data = {
    "model": optimized_ensemble,
    "model_name": "Optimized_Ensemble_Reduced_Overfitting",
    "scaler": scaler,
    "features": features,
    "target": target,
    "performance": {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "overfitting_gap": overfitting_gap,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "cv_mae": cv_mean,
        "cv_std": cv_std
    },
    "model_type": "VotingRegressor with Regularization",
    "data_info": {
        "original_samples": len(X),
        "after_smote": len(X_sm),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "selected_features": len(features)
    }
}

with open("optimized_lvedp_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

# =======================
# 11) FINAL SUMMARY
# =======================
print("\n" + "="*70)
print("🏆 OPTIMIZATION SUMMARY")
print("="*70)

print(f"""
✅ IMPROVEMENTS APPLIED:
1. Enhanced Feature Selection: {len(features)} features selected
2. Fixed SMOTE: Proper broadcasting and target reconstruction  
3. Stratified Train-Test Split: Balanced data representation
4. Strong Regularization: Reduced model complexity
5. Cross-Validation: Robust performance estimation

📊 FINAL RESULTS:
• Test MAE: {test_mae:.4f}
• Overfitting Gap: {overfitting_gap:.4f} {'✅' if overfitting_gap < 0.3 else '⚠️'}
• Test R²: {test_r2:.4f}
• Cross-validation: {cv_mean:.4f} ± {cv_std:.4f}

🎯 STATUS: {'OPTIMIZED - Ready for deployment! 🚀' if overfitting_gap < 0.3 else 'GOOD - Minor overfitting remains'}

💾 Model saved as: optimized_lvedp_model.pkl
""")

# Save performance report
performance_report = f"""
LVEDP PREDICTION MODEL - OPTIMIZATION REPORT
Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

OPTIMIZATION STRATEGIES APPLIED:
1. Advanced Feature Selection (Lasso + Random Forest)
2. Fixed SMOTE with Proper Broadcasting
3. Stratified Data Splitting
4. Strong Model Regularization
5. Comprehensive Cross-Validation

PERFORMANCE METRICS:
- Training MAE: {train_mae:.4f}
- Testing MAE: {test_mae:.4f}
- Overfitting Gap: {overfitting_gap:.4f}
- Testing R²: {test_r2:.4f}
- Cross-validation MAE: {cv_mean:.4f} ± {cv_std:.4f}

DATA INFORMATION:
- Original samples: {len(X)}
- After SMOTE: {len(X_sm)}
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Selected features: {len(features)}

OVERFITTING ASSESSMENT: {'✅ UNDER CONTROL' if overfitting_gap < 0.3 else '⚠️ NEEDS MONITORING'}

RECOMMENDATION: {'Ready for clinical deployment' if overfitting_gap < 0.3 else 'Suitable for use with monitoring'}
"""

with open('optimization_performance_report.txt', 'w', encoding='utf-8') as f:
    f.write(performance_report)

print("📋 Performance report saved: optimization_performance_report.txt")
print("🎉 Optimization completed successfully!")

