import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import optuna
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ============================================================
# Configuration
# ============================================================
PROCESSED_DIR = r'C:\Users\ccc\Desktop\algorithm\回归算法\02_特征提取\output'
OUTPUT_DIR = r'C:\Users\ccc\Desktop\algorithm\回归算法\05_DNN\output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load processed data
# ============================================================
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'processed_train.csv'))
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'processed_test.csv'))
selected_features = pd.read_csv(os.path.join(PROCESSED_DIR, 'selected_features.csv'))
feature_cols = selected_features['feature'].tolist()

X_train_full = train_df[feature_cols].values
y_train_full = train_df['Leak_pct'].values
X_test = test_df[feature_cols].values
y_test = test_df['Leak_pct'].values

# Split train into train (80%) and validation (20%) for Optuna
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f'Features: {feature_cols}')
print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

# ============================================================
# Optuna Bayesian Optimization
# ============================================================
print('\n' + '=' * 60)
print('DNN Optuna Hyperparameter Optimization (n_trials=50)')
print('=' * 60)

def create_model(n_layers, n_units, activation, dropout_rate, learning_rate):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    for _ in range(n_layers):
        model.add(layers.Dense(n_units, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))  # Regression output
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
    )
    return model

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    n_units = trial.suggest_categorical('n_units_per_layer', [16, 32, 64, 128])
    activation = trial.suggest_categorical('activation', ['relu', 'elu'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])

    model = create_model(n_layers, n_units, activation, dropout_rate, learning_rate)

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=15,
        restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    return val_loss

# Create Optuna study
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(),
    study_name='dnn_regression'
)

t0 = time.time()
study.optimize(objective, n_trials=50, show_progress_bar=True)
print(f'\nOptuna completed in {time.time() - t0:.1f}s')
print(f'Best trial: #{study.best_trial.number}')
print(f'Best params: {study.best_params}')
print(f'Best val MSE: {study.best_value:.4f}')

# Save Optuna results
optuna_df = study.trials_dataframe()
optuna_df.to_csv(os.path.join(OUTPUT_DIR, 'optuna_trials.csv'), index=False)

# ============================================================
# Final model training on full training set
# ============================================================
print('\n' + '=' * 60)
print('Final DNN Model Training (full training set)')
print('=' * 60)

best_params = study.best_params
final_model = create_model(
    n_layers=best_params['n_layers'],
    n_units=best_params['n_units_per_layer'],
    activation=best_params['activation'],
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate'],
)

final_model.summary()

early_stop = callbacks.EarlyStopping(
    monitor='loss', patience=15,
    restore_best_weights=True, verbose=1
)

history = final_model.fit(
    X_train_full, y_train_full,
    epochs=150,
    batch_size=best_params['batch_size'],
    callbacks=[early_stop],
    verbose=1,
    validation_split=0.0  # No validation split for final training
)

final_epochs = len(history.history['loss'])
print(f'Trained for {final_epochs} epochs (best at epoch {final_epochs - early_stop.patience if final_epochs > early_stop.patience else final_epochs})')

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(OUTPUT_DIR, 'dnn_training_history.csv'), index=False)

# ============================================================
# Final evaluation
# ============================================================
print('\n' + '=' * 60)
print('Final DNN Model Evaluation')
print('=' * 60)

y_train_pred = final_model.predict(X_train_full, verbose=0).flatten()
y_test_pred = final_model.predict(X_test, verbose=0).flatten()

def compute_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'SD': np.std(y_pred - y_true),
    }

train_metrics = compute_metrics(y_train_full, y_train_pred)
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
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'dnn_metrics.csv'), index=False)

# Engineering tolerance
within_tol = np.abs(y_test_pred - y_test) <= 5.0
tol_pct = within_tol.mean() * 100
print(f'\nEngineering tolerance (+/-5%): {tol_pct:.2f}% of test samples')

tolerance_df = pd.DataFrame({
    'Tolerance': ['+/-5%'],
    'Percentage': [f'{tol_pct:.2f}%'],
})
tolerance_df.to_csv(os.path.join(OUTPUT_DIR, 'dnn_tolerance.csv'), index=False)

# ============================================================
# Figure 1: True vs Predicted scatter plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_test_pred, alpha=0.7, edgecolors='white', linewidth=0.5, c='#FF9800')
lims = [min(y_test.min(), y_test_pred.min()) - 2, max(y_test.max(), y_test_pred.max()) + 2]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='y = x (ideal)')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel('True Leakage (%)', fontsize=12)
ax.set_ylabel('Predicted Leakage (%)', fontsize=12)
ax.set_title(f'DNN -- True vs Predicted (Test R2={test_metrics["R2"]:.4f})', fontsize=14)
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dnn_true_vs_pred.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Figure 1 saved: dnn_true_vs_pred.png')

# ============================================================
# Figure 2: Error distribution histogram
# ============================================================
errors = y_test_pred - y_test
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(errors, kde=True, bins=40, color='#FF9800', alpha=0.6, ax=ax)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Prediction Error (Predicted - True) [%]', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title(f'DNN -- Prediction Error Distribution (SD={test_metrics["SD"]:.4f})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dnn_error_dist.png'), dpi=300, bbox_inches='tight')
plt.close()
print('Figure 2 saved: dnn_error_dist.png')

print('\nDNN regression complete.')
