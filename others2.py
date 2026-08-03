import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from category_encoders import TargetEncoder
import scipy.stats as stats

# 1. LOAD THE TARGET SHEET
dataset_path = "/home/wandas/Desktop/G10/experiment/dataset_2.xlsx"

# FIXME: Replace 'Sheet1' with the actual name of the sheet containing the 800+ rows!
sheet_name_to_use = 'Sheet1' 

try:
    # We specify the sheet name explicitly here
    df = pd.read_excel(dataset_path, sheet_name=sheet_name_to_use)
    print(f"Successfully loaded sheet '{sheet_name_to_use}' with shape: {df.shape}")
except Exception as e:
    print(f"Error: {e}")
    exit()

# --- DYNAMIC HEADER ALIGNMENT ---
# If the first few rows contain empty title space, find where the questionnaire headers actually start
if 'TOTAL' not in df.columns and any(df.iloc[i].astype(str).str.contains('TOTAL').any() for i in range(min(10, len(df)))):
    for idx in range(min(10, len(df))):
        if df.iloc[idx].astype(str).str.contains('TOTAL').any():
            print(f"Fixing header alignment: Moving header to row index {idx + 1}")
            df = pd.read_excel(dataset_path, sheet_name=sheet_name_to_use, header=idx + 1)
            break

# Identify the target column (Looking for 'TOTAL' or the absolute final column)
target_col = 'TOTAL' if 'TOTAL' in df.columns else df.columns[-1]
print(f"Using target column: '{target_col}'")

# Clean and convert target values to continuous float metrics
df = df.dropna(subset=[target_col]).reset_index(drop=True)
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
df = df.dropna(subset=[target_col]).reset_index(drop=True)

# Feature and target separation
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify explicit feature datatypes
categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number", "int64", "float64"]).columns.tolist()

for col in categorical_cols:
    X[col] = X[col].astype(str)

# Handle missing data features via median values
for col in numeric_cols:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

print(f"Categorical features: {len(categorical_cols)} found")
print(f"Numeric features: {len(numeric_cols)} found")
print(f"Total valid survey rows for validation: {len(df)}")

if len(df) < 5:
    print("[ERROR] Insufficient valid dataset rows to execute validation split.")
    exit()

# 2. 5-Fold Cross-Validation Framework
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ols_r2, ols_mae, ols_rmse = [], [], []
lasso_r2, lasso_mae, lasso_rmse = [], [], []
chosen_lambdas = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    encoder = TargetEncoder(cols=categorical_cols, smoothing=10.0)
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_val_enc = encoder.transform(X_val)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_val_scaled = scaler.transform(X_val_enc)
    
    # Model 1: Baseline OLS
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    y_pred_ols = ols.predict(X_val_scaled)
    ols_r2.append(r2_score(y_val, y_pred_ols))
    ols_mae.append(mean_absolute_error(y_val, y_pred_ols))
    ols_rmse.append(root_mean_squared_error(y_val, y_pred_ols))
    
    # Model 2: TE-Lasso
    lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, max_iter=10000, random_state=42)
    lasso_cv.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_cv.predict(X_val_scaled)
    lasso_r2.append(r2_score(y_val, y_pred_lasso))
    lasso_mae.append(mean_absolute_error(y_val, y_pred_lasso))
    lasso_rmse.append(root_mean_squared_error(y_val, y_pred_lasso))
    chosen_lambdas.append(lasso_cv.alpha_)

stat_r2, p_val_r2 = stats.wilcoxon(lasso_r2, ols_r2)

print("\n" + "="*60)
print("             UPDATED SYSTEM DATA METRICS             ")
print("="*60)
print(f"Mean Lasso Alpha Parameter (λ): {np.mean(chosen_lambdas):.5f}")
print("-"*60)
print("BASELINE OLS INTERMEDIATE SCORE (Mean ± SD):")
print(f"  R² Score : {np.mean(ols_r2):.4f} ± {np.std(ols_r2):.4f}")
print(f"  MAE      : {np.mean(ols_mae):.4f} ± {np.std(ols_mae):.4f}")
print(f"  RMSE     : {np.mean(ols_rmse):.4f} ± {np.std(ols_rmse):.4f}")
print("-"*60)
print("PROPOSED TE-LASSO REGRESSION PIPELINE (Mean ± SD):")
print(f"  R² Score : {np.mean(lasso_r2):.4f} ± {np.std(lasso_r2):.4f}")
print(f"  MAE      : {np.mean(lasso_mae):.4f} ± {np.std(lasso_mae):.4f}")
print(f"  RMSE     : {np.mean(lasso_rmse):.4f} ± {np.std(lasso_rmse):.4f}")
print("-"*60)
print("CROSS-MODEL DISCRIMINANT VALUE:")
print(f"  Wilcoxon Test Score: {stat_r2}")
print(f"  Calculated p-value:  {p_val_r2:.5f}")
print("="*60)
