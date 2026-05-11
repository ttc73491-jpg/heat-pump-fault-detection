import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
DATA_DIR = r'C:\Users\ccc\Desktop\algorithm\回归算法\01_源数据'
OUTPUT_DIR = r'C:\Users\ccc\Desktop\algorithm\回归算法\02_特征提取\output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEAK_MAP = {
    'Heatpump_Leak_0pct.csv': 0,
    'Heatpump_Leak_5pct.csv': 5,
    'Heatpump_Leak_10pct.csv': 10,
    'Heatpump_Leak_20pct.csv': 20,
    'Heatpump_Leak_25pct.csv': 25,
    'Heatpump_Leak_30pct.csv': 30,
    'Heatpump_Leak_35pct.csv': 35,
    'Heatpump_Leak_40pct.csv': 40,
    'Heatpump_Leak_45pct.csv': 45,
    'Heatpump_Leak_50pct.csv': 50,
}

# Columns to delete by name (Stage 1 -- physical pruning)
DROP_COLS = [
    'V_sep_liq[%]',        # Gas-liquid separator level, not measurable in practice
    'h_suc[kJ/kg]',         # Enthalpy -- redundant
    'h_dis[kJ/kg]',
    'h_gc_mid[kJ/kg]',
    'h_gc_out[kJ/kg]',
    'h_eva_in[kJ/kg]',
    'h_eva_out[kJ/kg]',
    'T_amb[degC]',          # Ambient temperature
]

# ============================================================
# Stage 1: Load data & Physical pruning
# ============================================================
print('=' * 60)
print('STAGE 1: Data loading & physical pruning')
print('=' * 60)

dfs = []
for fname, leak_pct in LEAK_MAP.items():
    fpath = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(fpath)
    df['Leak_pct'] = leak_pct
    dfs.append(df)
    print(f'  Loaded {fname}: {len(df)} rows, leak={leak_pct}%')

df_all = pd.concat(dfs, ignore_index=True)
print(f'\nTotal samples: {len(df_all)}')
print(f'Original columns: {len(df_all.columns)} (1 time + {len(df_all.columns)-2} features + leak target)')

# Separate time, features, target
time_col = df_all['time[s]']
y = df_all['Leak_pct'].values
X = df_all.drop(columns=['time[s]', 'Leak_pct'])

# Physical pruning
print(f'\nDropping {len(DROP_COLS)} columns based on physical knowledge:')
for col in DROP_COLS:
    print(f'  - {col}')
X = X.drop(columns=DROP_COLS)
print(f'Remaining features: {X.shape[1]}')

feature_names_all = X.columns.tolist()

# ============================================================
# Stage 2: Train/Test Split
# ============================================================
print('\n' + '=' * 60)
print('STAGE 2: Train/test split (7:3)')
print('=' * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')

# ============================================================
# Stage 3: StandardScaler (fit only on training set)
# ============================================================
print('\n' + '=' * 60)
print('STAGE 3: StandardScaler normalization')
print('=' * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Scaler fitted on training set only -- no data leakage.')

# Convert to DataFrame for easier column handling
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names_all, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names_all, index=X_test.index)

# ============================================================
# Stage 4: Automated Feature Selection
# ============================================================

# ---- 4.1 Pearson correlation removal (|r| > 0.95) ----
print('\n' + '=' * 60)
print('STAGE 4.1: Pearson correlation deduplication (|r| > 0.95)')
print('=' * 60)

corr_matrix = X_train_scaled_df.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], r))

print(f'Found {len(high_corr_pairs)} highly correlated feature pairs:')
for f1, f2, r in high_corr_pairs:
    print(f'  {f1} <-> {f2}: r = {r:.4f}')

# Decide which feature to drop from each high-correlation pair
# Rule: drop the one with higher mean absolute correlation with all other features
cols_to_drop = set()
for f1, f2, _ in high_corr_pairs:
    if f1 in cols_to_drop or f2 in cols_to_drop:
        continue
    # Compute mean |r| with other features (excluding self)
    mean_corr_f1 = corr_matrix[f1].drop(f1).abs().mean()
    mean_corr_f2 = corr_matrix[f2].drop(f2).abs().mean()
    drop = f1 if mean_corr_f1 >= mean_corr_f2 else f2
    cols_to_drop.add(drop)
    keep = f1 if drop == f2 else f2
    reason = f'mean|r| = {mean_corr_f1:.4f}' if drop == f1 else f'mean|r| = {mean_corr_f2:.4f}'
    print(f'  -> Drop "{drop}" ({reason}), keep "{keep}"')

if cols_to_drop:
    X_train_scaled_df = X_train_scaled_df.drop(columns=list(cols_to_drop))
    X_test_scaled_df = X_test_scaled_df.drop(columns=list(cols_to_drop))
    print(f'\nDropped {len(cols_to_drop)} redundant features.')
    print(f'Remaining: {X_train_scaled_df.shape[1]} features')

# ---- 4.2 RF importance screening (90% cumulative threshold) ----
print('\n' + '=' * 60)
print('STAGE 4.2: RF importance screening (cumulative 90%)')
print('=' * 60)

rf_selector = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_scaled_df, y_train)

importances = rf_selector.feature_importances_
importance_df = pd.DataFrame({
    'feature': X_train_scaled_df.columns,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)
importance_df['cumulative'] = importance_df['importance'].cumsum()

# Select features at 90% cumulative threshold
selected_mask = importance_df['cumulative'] <= 0.90
# Ensure at least one feature is selected
n_selected = max(selected_mask.sum(), 1)
selected_features = importance_df['feature'].iloc[:n_selected].tolist()

print(f'Selected {len(selected_features)} / {len(importance_df)} features at 90% cumulative threshold:')
for _, row in importance_df.iterrows():
    marker = '[+]' if row['feature'] in selected_features else '[ ]'
    print(f'  {marker} {row["feature"]:30s}  importance={row["importance"]:.4f}  cumulative={row["cumulative"]:.4f}')

# Save feature importance
importance_df.to_csv(os.path.join(OUTPUT_DIR, 'rf_feature_importance.csv'), index=False)
print(f'\nFeature importance saved to output/rf_feature_importance.csv')

# ============================================================
# Final data reconstruction
# ============================================================
print('\n' + '=' * 60)
print('FINAL: Reconstruct scaled data with selected features only')
print('=' * 60)

X_train_final = X_train_scaled_df[selected_features].values
X_test_final = X_test_scaled_df[selected_features].values
print(f'Final training matrix: {X_train_final.shape}')
print(f'Final test matrix: {X_test_final.shape}')

# Save selected feature names
selected_df = pd.DataFrame({'feature': selected_features})
selected_df.to_csv(os.path.join(OUTPUT_DIR, 'selected_features.csv'), index=False)
print(f'Selected features saved to output/selected_features.csv')

# Save processed data (scaled, selected features only)
processed_train = pd.DataFrame(X_train_final, columns=selected_features)
processed_train['Leak_pct'] = y_train
processed_test = pd.DataFrame(X_test_final, columns=selected_features)
processed_test['Leak_pct'] = y_test

processed_train.to_csv(os.path.join(OUTPUT_DIR, 'processed_train.csv'), index=False)
processed_test.to_csv(os.path.join(OUTPUT_DIR, 'processed_test.csv'), index=False)
print(f'Processed data saved to output/processed_train.csv and processed_test.csv')

# Also save the scaler parameters for reuse
scaler_params = pd.DataFrame({
    'feature': feature_names_all,
    'mean': scaler.mean_,
    'scale': scaler.scale_
})
scaler_params.to_csv(os.path.join(OUTPUT_DIR, 'scaler_params.csv'), index=False)

# ============================================================
# Feature importance plot
# ============================================================
print('\n' + '=' * 60)
print('PLOT: Feature importance bar chart')
print('=' * 60)

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#2196F3' if f in selected_features else '#BDBDBD' for f in importance_df['feature']]
bars = ax.barh(range(len(importance_df)), importance_df['importance'], color=colors, edgecolor='white')
ax.set_yticks(range(len(importance_df)))
ax.set_yticklabels(importance_df['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (MSE reduction)', fontsize=12)
ax.set_title('Random Forest Feature Importance -- Regression (90% Cumulative Threshold)', fontsize=14)

# Add cumulative line
ax2 = ax.twiny()
ax2.plot(importance_df['cumulative'].values, range(len(importance_df)),
         'ro-', markersize=4, linewidth=1.5, label='Cumulative')
ax2.axvline(x=0.90, color='red', linestyle='--', linewidth=1, alpha=0.7, label='90% threshold')
ax2.set_xlabel('Cumulative Importance', fontsize=12)
ax2.legend(loc='lower right', fontsize=10)

# Legend for bar colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2196F3', label=f'Selected ({len(selected_features)} features)'),
    Patch(facecolor='#BDBDBD', label=f'Dropped ({len(importance_df) - len(selected_features)} features)'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Feature importance plot saved to output/feature_importance.png')

# Correlation heatmap of selected features
fig, ax = plt.subplots(figsize=(14, 12))
corr_selected = X_train_scaled_df[selected_features].corr()
sns.heatmap(corr_selected, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            annot_kws={'size': 7}, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Pearson Correlation Matrix — Selected Features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Correlation heatmap saved to output/correlation_heatmap.png')

print('\n' + '=' * 60)
print('FEATURE EXTRACTION COMPLETE')
print('=' * 60)
