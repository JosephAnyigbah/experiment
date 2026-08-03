import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from category_encoders import TargetEncoder
import scipy.stats as stats

# 1. Load Excel dataset
dataset_path = "/home/wandas/Desktop/G10/experiment/dataset_2.xlsx"

try:
    df = pd.read_excel(dataset_path)
    print(f"Successfully loaded dataset with shape: {df.shape}")
except ModuleNotFoundError:
    print("\n[ERROR] Missing Excel dependency. Please run: pip install openpyxl\n")
    raise

# Ensure all column names are uniformly strings
df.columns = df.columns.astype(str)

target_col = 'Unnamed: 14'
print(f"Using target column: '{target_col}'")

# --- STEP 1: Clean Target Variable ---
df = df.dropna(subset=[target_col]).reset_index(drop=True)
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# Separate features and target
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# --- STEP 2: Detect & Clean Feature Types ---
categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number", "int64", "float64"]).columns.tolist()

# Ensure numeric columns are forced to numeric (coercing invalid non-numeric strings to NaN)
for col in numeric_cols:
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Ensure categorical columns are uniformly strings
for col in categorical_cols:
    X[col] = X[col].astype(str)

print(f"Categorical features detected: {categorical_cols}")
print(f"Numeric features detected: {numeric_cols}")
print(f"Total valid samples remaining for cross-validation: {len(df)}")

# =====================================================================
# 3. LEAKAGE-FREE 5-FOLD CROSS-VALIDATION LOOP
# =====================================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

ols_r2, ols_mae, ols_rmse = [], [], []
lasso_r2, lasso_mae, lasso_rmse = [], [], []
chosen_lambdas = []

smoothing_val = 10.0

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # 1. Target Encoding inside the fold
    encoder = TargetEncoder(cols=categorical_cols, smoothing=smoothing_val, handle_missing='value', handle_unknown='value')
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_val_enc = encoder.transform(X_val)
    
    # Preserve string column names
    X_train_enc.columns = X_train_enc.columns.astype(str)
    X_val_enc.columns = X_val_enc.columns.astype(str)
    
    # 2. Impute any remaining NaNs (numerical or post-encoding missing values)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_enc)
    X_val_imp = imputer.transform(X_val_enc)
    
    # 3. Continuous Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    
    # --- MODEL A: Baseline OLS Linear Regression ---
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    y_pred_ols = ols.predict(X_val_scaled)
    
    ols_r2.append(r2_score(y_val, y_pred_ols))
    ols_mae.append(mean_absolute_error(y_val, y_pred_ols))
    ols_rmse.append(root_mean_squared_error(y_val, y_pred_ols))
    
    # --- MODEL B: Proposed Lasso CV Pipeline ---
    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=20000, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_cv.predict(X_val_scaled)
    
    lasso_r2.append(r2_score(y_val, y_pred_lasso))
    lasso_mae.append(mean_absolute_error(y_val, y_pred_lasso))
    lasso_rmse.append(root_mean_squared_error(y_val, y_pred_lasso))
    chosen_lambdas.append(lasso_cv.alpha_)

# =====================================================================
# 4. STATISTICAL VALIDATION
# =====================================================================
stat_r2, p_val_r2 = stats.wilcoxon(lasso_r2, ols_r2)

# =====================================================================
# 5. GENERATE CLEAN OUTPUT
# =====================================================================
print("\n" + "="*60)
print("             EXACT MANUSCRIPT PERFORMANCE METRICS             ")
print("="*60)
print(f"Target Encoding Smoothing Parameter (α): {smoothing_val}")
print(f"Mean Optimized Lasso Regularization Scale (λ): {np.mean(chosen_lambdas):.5f}")
print("-"*60)
print("BASELINE UNREGULARIZED OLS PERFORMANCE (Mean ± SD):")
print(f"  R² Score : {np.mean(ols_r2):.4f} ± {np.std(ols_r2):.4f}")
print(f"  MAE      : {np.mean(ols_mae):.4f} ± {np.std(ols_mae):.4f}")
print(f"  RMSE     : {np.mean(ols_rmse):.4f} ± {np.std(ols_rmse):.4f}")
print("-"*60)
print("TE-LASSO PROPOSED FRAMEWORK PERFORMANCE (Mean ± SD):")
print(f"  R² Score : {np.mean(lasso_r2):.4f} ± {np.std(lasso_r2):.4f}")
print(f"  MAE      : {np.mean(lasso_mae):.4f} ± {np.std(lasso_mae):.4f}")
print(f"  RMSE     : {np.mean(lasso_rmse):.4f} ± {np.std(lasso_rmse):.4f}")
print("-"*60)
print("STATISTICAL VALIDATION PARAMETERS:")
print(f"  Wilcoxon Signed-Rank Test Statistic: {stat_r2}")
print(f"  Asymptotic p-value (R² comparison):  {p_val_r2:.5f}")
print("="*60)
