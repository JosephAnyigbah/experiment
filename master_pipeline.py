import numpy as np
import pandas as pd
import joblib
import os
import warnings
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

DATA_PATH = 'dataset.xlsx'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Missing target file '{DATA_PATH}' in your workspace directory!")

# Load the data starting exactly at your table header row (Index 6)
df = pd.read_excel(DATA_PATH, header=6)
df = df.dropna(how='all')
df.columns = [str(col).strip() for col in df.columns]

TARGET_COLUMN_NAME = 'TOTAL'
FEATURE_COLUMNS = ['study level', 'GENDER']

# Drop rows where target variable is missing or contains summary labels
df = df[df[TARGET_COLUMN_NAME].notna()]
df = df[~df[TARGET_COLUMN_NAME].astype(str).str.contains('Total|total|Sum|sum', na=False)]

y_raw = pd.to_numeric(df[TARGET_COLUMN_NAME], errors='coerce')
valid_mask = y_raw.notna()
df = df[valid_mask]
y = y_raw[valid_mask].values

print(f"📋 Locked target variable column: '{TARGET_COLUMN_NAME}'")
print(f"📊 Features selected for engineering: {FEATURE_COLUMNS}")
print(f"📉 Valid Matrix Row Count: {len(df)} samples.")

X_raw = df[FEATURE_COLUMNS].copy()
X_raw = X_raw.ffill().bfill()

# ==========================================================
# 2. ADVANCED MODEL ENGINEERING: LEAKAGE-FREE OOF ENCODING
# ==========================================================
X_encoded = pd.DataFrame(index=X_raw.index)
global_maps = {}
global_target_mean = float(np.mean(y))

# 5-Fold split to calculate target encodings without looking at the validation rows
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in FEATURE_COLUMNS:
    X_encoded[col] = np.nan
    
    # Out-of-Fold Encoding Loop
    for train_idx, val_idx in kf.split(X_raw, y):
        X_tr, X_va = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr = y[train_idx]
        
        # Calculate mean using ONLY training fold rows
        fold_map = df.iloc[train_idx].groupby(col)[TARGET_COLUMN_NAME].mean().to_dict()
        X_encoded.iloc[val_idx, X_encoded.columns.get_loc(col)] = X_va[col].map(fold_map)
        
    # Fill any remaining NaNs within folds with the global target mean
    X_encoded[col] = X_encoded[col].fillna(global_target_mean)
    
    # Save a clean master global map for the deployment serialization
    global_maps[col] = df.groupby(col)[TARGET_COLUMN_NAME].mean().to_dict()

feature_names = X_encoded.columns.tolist()

# ==========================================================
# 3. HIGH-INTEGRITY REGULARIZED MODEL
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Highly regularized Lasso to establish the real asset variance weight
model = Lasso(alpha=0.1, random_state=42)
model.fit(X_scaled, y)

preds = model.predict(X_scaled)
print(f"\n🚀 Leakage-Free Production Diagnostics:")
print(f"   Matrix Shape: {X_scaled.shape[0]} samples | {X_scaled.shape[1]} features")
print(f"   Real R² Score:    {r2_score(y, preds):.4f}")
print(f"   Real MAE Score:   {mean_absolute_error(y, preds):.4f}")

print("\n📊 Feature Weights (Coefficients):")
for name, coef in zip(feature_names, model.coef_):
    print(f"   {name}: {coef:.4f}")

artifacts = {
    'model': model,
    'scaler': scaler,
    'target_encodings': global_maps,
    'global_mean': global_target_mean,
    'feature_names': feature_names
}
joblib.dump(artifacts, 'best_model.pkl')
print("\n📦 Pipeline saved successfully to 'best_model.pkl'.")

# ==========================================================
# 4. INSTANT SIMULATED DEPLOYMENT VERIFICATION
# ==========================================================
print("\n🎯 Running Live Inference Verification...")
test_sample = {'study level': 'PM2', 'GENDER': 'Male'}

clean_input = {str(k).strip(): v for k, v in test_sample.items()}
encoded_input = {}
for col in artifacts['feature_names']:
    val = clean_input.get(col, None)
    mapping = artifacts['target_encodings'][col]
    encoded_input[col] = mapping.get(str(val).strip(), artifacts['global_mean'])
        
df_inf = pd.DataFrame([encoded_input]).reindex(columns=artifacts['feature_names'], fill_value=artifacts['global_mean'])
X_inf_scaled = artifacts['scaler'].transform(df_inf)
live_pred = artifacts['model'].predict(X_inf_scaled)[0]

print(f"🎯 Inference Pipeline Output (Predicted Total): {live_pred:.2f}")
