import numpy as np
import pandas as pd
import joblib
import os
import warnings
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

DATA_PATH = 'dataset_2.xlsx'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing target file '{DATA_PATH}' in directory.")

# ==========================================================
# 1. DYNAMIC HEADER DETECTION (No more hardcoded row skips)
# ==========================================================
# Read raw without header to find where the table actually starts
df_raw = pd.read_excel(DATA_PATH, header=None)

header_idx = 0
for idx, row in df_raw.iterrows():
    row_str = [str(x).lower() for x in row.dropna()]
    # Look for your known structural anchors like 'gender' or 'study level'
    if any('gender' in s or 'study' in s or 'level' in s for s in row_str):
        header_idx = idx
        break

print(f"🎯 Dynamically detected table header at Excel row index: {header_idx}")

# Reload data with the correct dynamic header position
df = pd.read_excel(DATA_PATH, header=header_idx)
df = df.dropna(how='all')

# Strip column names immediately
df.columns = [str(col).strip() for col in df.columns]

# Standard target selection (last column)
target_column = df.columns[-1]
df = df[df[target_column].notna()]
df = df[~df[target_column].astype(str).str.contains('Total|total|Sum|sum', na=False)]

y = pd.to_numeric(df[target_column], errors='coerce')
# Drop rows where target conversion failed
valid_mask = y.notna()
df = df[valid_mask]
y = y[valid_mask].values

X_raw = df.drop(columns=[target_column], errors='ignore')

# Prune tracking ids, metadata, aggregates, and alignment artifacts
id_cols = [c for c in X_raw.columns if any(x in c.lower() for x in ['id', 'name', 'total', 'unnamed', '72', 'seminary'])]
X_raw = X_raw.drop(columns=id_cols)
X_raw = X_raw.ffill().bfill()

# ==========================================================
# 2. DENSE TARGET ENCODING LOGIC
# ==========================================================
target_encodings = {}
X_encoded = pd.DataFrame(index=X_raw.index)

for col in X_raw.columns:
    if X_raw[col].dtype == 'object' or isinstance(X_raw[col].dtype, pd.CategoricalDtype):
        mapping = df.groupby(col)[target_column].mean().to_dict()
        target_encodings[col] = mapping
        X_encoded[col] = X_raw[col].map(mapping).fillna(np.mean(y))
    else:
        X_encoded[col] = pd.to_numeric(X_raw[col], errors='coerce').fillna(np.mean(y))

feature_names = X_encoded.columns.tolist()
global_target_mean = float(np.mean(y))

# ==========================================================
# 3. PRODUCTION TRAIN & EVALUATE
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

model = Lasso(alpha=0.01, random_state=42)
model.fit(X_scaled, y)

preds = model.predict(X_scaled)
print(f"\n🚀 Leakage-Free Production Diagnostics:")
print(f"   Matrix Shape: {X_scaled.shape[0]} samples | {X_scaled.shape[1]} features")
print(f"   R² Score:    {r2_score(y, preds):.4f}")
print(f"   MAE Score:   {mean_absolute_error(y, preds):.4f}")

print("\n📊 Feature Weights (Coefficients):")
for name, coef in zip(feature_names, model.coef_):
    print(f"   {name}: {coef:.4f}")

artifacts = {
    'model': model,
    'scaler': scaler,
    'target_encodings': target_encodings,
    'global_mean': global_target_mean,
    'feature_names': feature_names
}
joblib.dump(artifacts, 'best_model.pkl')
print("\n📦 Pipeline saved successfully to 'best_model.pkl'.")
