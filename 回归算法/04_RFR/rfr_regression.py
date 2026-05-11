import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
PROCESSED_DIR = r'C:\Users\ccc\Desktop\algorithm\回归算法\02_特征提取\output'
OUTPUT_DIR = r'C:\Users\ccc\Desktop\algorithm\回归算法\04_RFR\output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load processed data
# ============================================================
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'processed_train.csv'))
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'processed_test.csv'))
selected_features = pd.read_csv(os.path.join(PROCESSED_DIR, 'selected_features.csv'))
feature_cols = selected_features['feature'].tolist()

X_train = train_df[feature_cols].values
y_train = train_df['Leak_pct'].values
X_test = test_df[feature_cols].values
y_test = test_df['Leak_pct'].values

print(f'Features: {feature_cols}')
print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# ============================================================
# GridSearchCV for RFR hyperparameters
# ============================================================
print('\n' + '=' * 60)
print('RFR Grid Search')
print('=' * 60)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 20],
}

rfr = RandomForestRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(
    rfr, param_grid,
    cv=5, scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=1
)

t0 = time.time()
grid.fit(X_train, y_train)
print(f'Grid search completed in {time.time() - t0:.1f}s')
print(f'Best params: {grid.best_params_}')
print(f'Best CV MSE: {-grid.best_score_:.4f}')

# ============================================================
# Final model training & evaluation
# ============================================================
print('\n' + '=' * 60)
print('Final RFR Model Evaluation')
print('=' * 60)

best_rfr = grid.best_estimator_

y_train_pred = best_rfr.predict(X_train)
y_test_pred = best_rfr.predict(X_test)

# Compute metrics
def compute_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'SD': np.std(y_pred - y_true),
    }

train_metrics = compute_metrics(y_train, y_train_pred)
test_metrics = compute_metrics(y_test, y_test_pred)

print('\n| Metric | Training Set | Test Set |')
print('| :--- | :--- | :--- |')
for metric in ['R2', 'RMSE', 'MAE', 'SD']:
    print(f'| {metric} | {train_metrics[metric]:.4f} | {test_metrics[metric]:.4f} |')

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['R2', 'RMSE', 'MAE', 'SD'],
    'Training_Set': [train_metrics[m] for m in ['R2', 'RMSE', 'MAE', 'SD']],
    'Test_Set': [test_metrics[m] for m in ['R2', 'RMSE', 'MAE', 'SD']],
})
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'rfr_metrics.csv'), index=False)

# Engineering tolerance
within_tol = np.abs(y_test_pred - y_test) <= 5.0
tol_pct = within_tol.mean() * 100
print(f'\nEngineering tolerance (+/-5%): {tol_pct:.2f}% of test samples')

tolerance_df = pd.DataFrame({
    'Tolerance': ['+/-5%'],
    'Percentage': [f'{tol_pct:.2f}%'],
})
tolerance_df.to_csv(os.path.join(OUTPUT_DIR, 'rfr_tolerance.csv'), index=False)

# Feature importance
print('\nTop 3 feature contributions (MSE reduction):')
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_rfr.feature_importances_
}).sort_values('importance', ascending=False)
for i, (_, row) in enumerate(feature_imp.head(3).iterrows()):
    print(f'  {i+1}. {row["feature"]}: {row["importance"]:.4f}')
feature_imp.to_csv(os.path.join(OUTPUT_DIR, 'rfr_feature_importance.csv'), index=False)

# ============================================================
# Figure 1: True vs Predicted scatter plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_test_pred, alpha=0.7, edgecolors='white', linewidth=0.5, c='#4CAF50')
lims = [min(y_test.min(), y_test_pred.min()) - 2, max(y_test.max(), y_test_pred.max()) + 2]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='y = x (ideal)')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('True Leakage (%)', fontsize=12)
ax.set_ylabel('Predicted Leakage (%)', fontsize=12)
ax.set_title(f'RFR -- True vs Predicted (Test R2={test_metrics["R2"]:.4f})', fontsize=14)
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rfr_true_vs_pred.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Figure 1 saved: rfr_true_vs_pred.png')

# ============================================================
# Figure 2: Error distribution histogram
# ============================================================
errors = y_test_pred - y_test
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=40, color='#4CAF50', alpha=0.6, ax=ax)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Prediction Error (Predicted - True) [%]', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'RFR -- Prediction Error Distribution (SD={test_metrics["SD"]:.4f})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rfr_error_dist.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Figure 2 saved: rfr_error_dist.png')

print('\nRFR regression complete.')
