"""
DNN时序建模 — 滑动窗口 + 1D-CNN/LSTM
利用仿真数据的时序特性，提取动态模式辅助故障诊断
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Input, Dropout, BatchNormalization, LeakyReLU,
                                      Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                      LSTM, Add, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import warnings
import os
import time

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(42)
tf.random.set_seed(42)

data_dir = r'C:\Users\ccc\Desktop\algorithm\01_源数据'
output_dir = r'C:\Users\ccc\Desktop\algorithm\05_深度神经网络\architecture_sweep'
os.makedirs(output_dir, exist_ok=True)

# ========== 配置 ==========
SELECTED_FEATURES = [
    'h_gc_mid[kJ/kg]', 'P_dis[bar]', 'h_dis[kJ/kg]', 'P_gc_out[bar]',
    'Q_heat_s2[kW]', 'Q_heat_s1[kW]', 'T_mid[degC]', 'h_eva_in[kJ/kg]',
    'T_air_in[degC]', 'P_gc_mid[bar]', 'W_comp[kW]', 'T_eva_out[degC]',
    'Q_heat_total[kW]', 'P_eva_out[bar]', 'h_gc_out[kJ/kg]', 'P_eva_in[bar]',
    'P_suc[bar]', 'T_gc_mid[degC]', 'T_eva_in[degC]', 'T_suc[degC]',
]

FILE_CONDITIONS = {
    'Heatpump_Leak_0pct.csv':  {'severity': 0, 'original_label': 0},
    'Heatpump_Leak_5pct.csv':  {'severity': 1, 'original_label': 1},
    'Heatpump_Leak_10pct.csv': {'severity': 1, 'original_label': 2},
    'Heatpump_Leak_20pct.csv': {'severity': 1, 'original_label': 3},
    'Heatpump_Leak_25pct.csv': {'severity': 2, 'original_label': 4},
    'Heatpump_Leak_30pct.csv': {'severity': 2, 'original_label': 5},
    'Heatpump_Leak_35pct.csv': {'severity': 2, 'original_label': 6},
    'Heatpump_Leak_40pct.csv': {'severity': 3, 'original_label': 7},
    'Heatpump_Leak_45pct.csv': {'severity': 3, 'original_label': 8},
    'Heatpump_Leak_50pct.csv': {'severity': 3, 'original_label': 9},
}

SEVERITY_NAMES = ['Normal(0%)', 'Light(5-20%)', 'Medium(25-35%)', 'Severe(40-50%)']
GROUPS = {
    0: {'name': 'Normal', 'labels': [0]},
    1: {'name': 'Light',  'labels': [1, 2, 3]},
    2: {'name': 'Medium', 'labels': [4, 5, 6]},
    3: {'name': 'Severe', 'labels': [7, 8, 9]},
}

WINDOW_SIZE = 20
STRIDE = 5
N_FEATURES = len(SELECTED_FEATURES)


def create_sliding_windows(data, labels_orig, labels_sev, window_size, stride):
    """从时序数据创建滑动窗口样本"""
    X_windows, y_orig_windows, y_sev_windows = [], [], []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        X_windows.append(data[start:start + window_size])
        y_orig_windows.append(labels_orig[start])
        y_sev_windows.append(labels_sev[start])
    return np.array(X_windows), np.array(y_orig_windows), np.array(y_sev_windows)


def deduplicate_by_time(df):
    """按时间步去重（取平均），同一时间步的多行合并为一行"""
    # 第一列是 '#', 第二列是 'time[s]'
    time_col = df.columns[1]
    feature_cols = [c for c in df.columns if c not in [df.columns[0], time_col]]
    # 按time分组取均值
    grouped = df.groupby(time_col)[feature_cols].mean().reset_index()
    return grouped


def calculate_metrics(y_true, y_pred, n_classes):
    recall_per_class = np.zeros(n_classes)
    far_per_class = np.zeros(n_classes)
    gma_per_class = np.zeros(n_classes)
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        recall_per_class[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        far_per_class[i] = FP / (FP + TN) if (FP + TN) > 0 else 0
        specificity = 1 - far_per_class[i]
        gma_per_class[i] = np.sqrt(recall_per_class[i] * specificity)
    return recall_per_class, far_per_class, gma_per_class


def build_cnn(n_classes, name='cnn'):
    """1D-CNN 时序模型"""
    inputs = Input(shape=(WINDOW_SIZE, N_FEATURES), name=f'{name}_input')

    x = Conv1D(64, 5, padding='same', kernel_regularizer=l2(1e-4), name=f'{name}_c1')(inputs)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = LeakyReLU(alpha=0.1, name=f'{name}_act1')(x)
    x = MaxPooling1D(2, name=f'{name}_mp1')(x)
    x = Dropout(0.2, name=f'{name}_do1')(x)

    x = Conv1D(128, 3, padding='same', kernel_regularizer=l2(1e-4), name=f'{name}_c2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = LeakyReLU(alpha=0.1, name=f'{name}_act2')(x)
    x = MaxPooling1D(2, name=f'{name}_mp2')(x)
    x = Dropout(0.2, name=f'{name}_do2')(x)

    x = Conv1D(256, 3, padding='same', kernel_regularizer=l2(1e-4), name=f'{name}_c3')(x)
    x = BatchNormalization(name=f'{name}_bn3')(x)
    x = LeakyReLU(alpha=0.1, name=f'{name}_act3')(x)
    x = GlobalAveragePooling1D(name=f'{name}_gap')(x)

    x = Dense(128, kernel_regularizer=l2(1e-4), name=f'{name}_d1')(x)
    x = BatchNormalization(name=f'{name}_bn4')(x)
    x = LeakyReLU(alpha=0.1, name=f'{name}_act4')(x)
    x = Dropout(0.3, name=f'{name}_do3')(x)

    x = Dense(64, kernel_regularizer=l2(1e-4), name=f'{name}_d2')(x)
    x = BatchNormalization(name=f'{name}_bn5')(x)
    x = LeakyReLU(alpha=0.1, name=f'{name}_act5')(x)
    x = Dropout(0.3, name=f'{name}_do4')(x)

    outputs = Dense(n_classes, activation='softmax', name=f'{name}_output')(x)
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_lstm(n_classes, name='lstm'):
    """LSTM 时序模型"""
    inputs = Input(shape=(WINDOW_SIZE, N_FEATURES), name=f'{name}_input')

    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4), name=f'{name}_l1')(inputs)
    x = BatchNormalization(name=f'{name}_bn1')(x)
    x = Dropout(0.3, name=f'{name}_do1')(x)

    x = LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4), name=f'{name}_l2')(x)
    x = BatchNormalization(name=f'{name}_bn2')(x)
    x = Dropout(0.3, name=f'{name}_do2')(x)

    x = Dense(64, kernel_regularizer=l2(1e-4), name=f'{name}_d1')(x)
    x = BatchNormalization(name=f'{name}_bn3')(x)
    x = LeakyReLU(alpha=0.1, name=f'{name}_act1')(x)
    x = Dropout(0.3, name=f'{name}_do3')(x)

    outputs = Dense(n_classes, activation='softmax', name=f'{name}_output')(x)
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_cm(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_norm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cm_norm[i,j]:.1%}\n({cm[i,j]})',
                    ha="center", va="center",
                    color="white" if cm_norm[i,j] > thresh else "black", fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()


def callbacks_fn():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0),
    ]


# ==================== 数据加载与窗口化 ====================
print("=" * 70)
print("DNN 时序建模 — 滑动窗口 + 1D-CNN/LSTM")
print("=" * 70)

all_data = []
all_labels_orig = []
all_labels_sev = []

for filename, info in FILE_CONDITIONS.items():
    df = pd.read_csv(os.path.join(data_dir, filename))
    # 去重：按时间步取均值
    df_dedup = deduplicate_by_time(df)
    features = df_dedup[SELECTED_FEATURES].values
    n_rows = len(features)
    all_data.append(features)
    all_labels_orig.append(np.full(n_rows, info['original_label']))
    all_labels_sev.append(np.full(n_rows, info['severity']))
    print(f"  {filename}: {len(df)} rows -> {n_rows} unique time steps")

# 合并所有数据
X_all = np.vstack(all_data)
y_orig_all = np.concatenate(all_labels_orig)
y_sev_all = np.concatenate(all_labels_sev)

print(f"\n总时间步: {len(X_all)}")

# 按文件分别做滑动窗口（保持文件内时序连续性）
X_windows_list, y_orig_win_list, y_sev_win_list = [], [], []

start_idx = 0
for filename, info in FILE_CONDITIONS.items():
    df = pd.read_csv(os.path.join(data_dir, filename))
    df_dedup = deduplicate_by_time(df)
    n_rows = len(df_dedup)
    file_data = X_all[start_idx:start_idx + n_rows]
    file_orig = y_orig_all[start_idx:start_idx + n_rows]
    file_sev = y_sev_all[start_idx:start_idx + n_rows]

    X_w, y_ow, y_sw = create_sliding_windows(
        file_data, file_orig, file_sev, WINDOW_SIZE, STRIDE
    )
    X_windows_list.append(X_w)
    y_orig_win_list.append(y_ow)
    y_sev_win_list.append(y_sw)
    start_idx += n_rows

X_windows = np.concatenate(X_windows_list, axis=0)
y_orig_windows = np.concatenate(y_orig_win_list)
y_sev_windows = np.concatenate(y_sev_win_list)

print(f"滑动窗口: {len(X_windows)} (W={WINDOW_SIZE}, stride={STRIDE})")
print(f"  每文件约 {len(X_windows)//10} 个窗口")

# 标准化：对每个窗口内做特征级标准化（跨所有窗口计算统计量）
# 先 flatten -> fit scaler -> reshape
n_windows, w_size, n_feat = X_windows.shape
X_flat = X_windows.reshape(-1, n_feat)
scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat)
X_windows_scaled = X_flat_scaled.reshape(n_windows, w_size, n_feat)

# 7:3 split（stratify by severity group）
X_train, X_test, y_train_sev, y_test_sev, y_train_orig, y_test_orig = train_test_split(
    X_windows_scaled, y_sev_windows, y_orig_windows,
    test_size=0.3, random_state=42, stratify=y_sev_windows
)
print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

# 从训练集再分出验证集
X_train_sub, X_val, y_train_sub_sev, y_val_sev = train_test_split(
    X_train, y_train_sev, test_size=0.1, random_state=42, stratify=y_train_sev
)

# ==================== 实验 1: CNN-1D 扁平 10 分类 ====================
print("\n" + "=" * 70)
print("实验 1: CNN-1D 扁平 10 分类")
print("=" * 70)

y_train_sub_oh = to_categorical(y_train_orig[np.isin(y_train_sev, np.arange(10))], 10)
# 需要对应训练集的 original labels
_, _, _, _, y_train_sub_orig, _ = train_test_split(
    X_train, y_train_sev, y_train_orig, test_size=0.1, random_state=42, stratify=y_train_sev
)
y_train_sub_oh = to_categorical(y_train_sub_orig, 10)
y_val_oh = to_categorical(y_val_sev, 10)  # 验证集用severity做近似（4类不够，需要orig）
# 重新构建验证集的orig
X_train_s, X_val_s, y_train_s_sev, y_val_s_sev, y_train_s_orig, y_val_s_orig = train_test_split(
    X_train, y_train_sev, y_train_orig, test_size=0.1, random_state=42, stratify=y_train_sev
)
y_train_s_oh = to_categorical(y_train_s_orig, 10)
y_val_s_oh = to_categorical(y_val_s_orig, 10)

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model_cnn10 = build_cnn(10, 'cnn_flat10')
print(f"CNN-1D 参数量: {model_cnn10.count_params():,}")

t0 = time.time()
hist_cnn10 = model_cnn10.fit(
    X_train_s, y_train_s_oh,
    epochs=300, batch_size=64,
    validation_data=(X_val_s, y_val_s_oh),
    callbacks=callbacks_fn(), verbose=2
)
t1 = time.time()

y_pred_cnn10 = np.argmax(model_cnn10.predict(X_test, verbose=0), axis=1)
acc_cnn10 = accuracy_score(y_test_orig, y_pred_cnn10)
rec_cnn10, far_cnn10, gma_cnn10 = calculate_metrics(y_test_orig, y_pred_cnn10, 10)

print(f"\nCNN-1D Flat 10-class Accuracy: {acc_cnn10*100:.2f}% | Time: {t1-t0:.1f}s")
print(f"Macro Recall: {np.mean(rec_cnn10)*100:.2f}% | Macro GMA: {np.mean(gma_cnn10)*100:.2f}%")
for i in range(10):
    print(f"  Class {i}: Recall={rec_cnn10[i]*100:6.2f}% FAR={far_cnn10[i]*100:6.2f}% GMA={gma_cnn10[i]*100:6.2f}%")

plot_cm(confusion_matrix(y_test_orig, y_pred_cnn10),
        [f'{i}' for i in range(10)],
        'CNN-1D Flat 10-Class', 'ts_cnn_flat10_cm.png')

# ==================== 实验 2: LSTM 扁平 10 分类 ====================
print("\n" + "=" * 70)
print("实验 2: LSTM 扁平 10 分类")
print("=" * 70)

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model_lstm10 = build_lstm(10, 'lstm_flat10')
print(f"LSTM 参数量: {model_lstm10.count_params():,}")

t0 = time.time()
hist_lstm10 = model_lstm10.fit(
    X_train_s, y_train_s_oh,
    epochs=300, batch_size=64,
    validation_data=(X_val_s, y_val_s_oh),
    callbacks=callbacks_fn(), verbose=2
)
t1 = time.time()

y_pred_lstm10 = np.argmax(model_lstm10.predict(X_test, verbose=0), axis=1)
acc_lstm10 = accuracy_score(y_test_orig, y_pred_lstm10)
rec_lstm10, far_lstm10, gma_lstm10 = calculate_metrics(y_test_orig, y_pred_lstm10, 10)

print(f"\nLSTM Flat 10-class Accuracy: {acc_lstm10*100:.2f}% | Time: {t1-t0:.1f}s")
print(f"Macro Recall: {np.mean(rec_lstm10)*100:.2f}% | Macro GMA: {np.mean(gma_lstm10)*100:.2f}%")

# ==================== 实验 3: CNN-1D 分层分类 ====================
print("\n" + "=" * 70)
print("实验 3: CNN-1D 分层分类 (Stage 1: 4-class → Stage 2: 组内细分)")
print("=" * 70)

# Stage 1: 4-class severity
print("\n--- Stage 1: 4-Class Severity ---")
y_train_s1_oh = to_categorical(y_train_s_sev, 4)
y_val_s1_oh = to_categorical(y_val_s_sev, 4)

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model_s1 = build_cnn(4, 'stage1_cnn')
print(f"Stage 1 CNN 参数量: {model_s1.count_params():,}")

t0 = time.time()
hist_s1 = model_s1.fit(
    X_train_s, y_train_s1_oh,
    epochs=300, batch_size=64,
    validation_data=(X_val_s, y_val_s1_oh),
    callbacks=callbacks_fn(), verbose=2
)
t1 = time.time()

y_pred_s1 = np.argmax(model_s1.predict(X_test, verbose=0), axis=1)
acc_s1 = accuracy_score(y_test_sev, y_pred_s1)
rec_s1, far_s1, gma_s1 = calculate_metrics(y_test_sev, y_pred_s1, 4)

print(f"\nStage 1 Accuracy: {acc_s1*100:.2f}% | Time: {t1-t0:.1f}s")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:>6.2f}% FAR={far_s1[i]*100:>6.2f}% GMA={gma_s1[i]*100:>6.2f}%")

plot_cm(confusion_matrix(y_test_sev, y_pred_s1), SEVERITY_NAMES,
        'CNN-1D Stage 1: 4-Class Severity', 'ts_cnn_stage1_cm.png')

# Stage 2: 组内细分（复用静态DNN模型，因为Stage 2用静态特征就够了）
print("\n--- Stage 2: 组内细分 (复用静态DNN Stage 2模型) ---")
# 从X_test还原静态特征：取窗口最后一行（最新时间步）
X_test_static = X_test[:, -1, :]  # (n_windows, n_features)

overall_correct = 0
overall_total = 0
group_results = {}

for group_id in range(4):
    group_labels = GROUPS[group_id]['labels']
    print(f"\n{GROUPS[group_id]['name']} Group (labels: {group_labels}):")

    if group_id == 0:
        mask = (y_test_sev == group_id)
        if np.sum(mask) > 0:
            correct = np.sum(y_pred_s1[mask] == y_test_sev[mask])
            overall_correct += correct
            overall_total += np.sum(mask)
            group_results[0] = {'correct': correct, 'total': np.sum(mask), 'acc': 100.0}
            print(f"  {np.sum(mask)} samples, all correct (no sub-classifier needed)")
        continue

    correct_mask_s1 = (y_pred_s1 == y_test_sev)
    test_mask = (y_test_sev == group_id) & correct_mask_s1
    X_test_g = X_test_static[test_mask]
    y_test_g = y_test_orig[test_mask]

    if len(X_test_g) == 0:
        group_results[group_id] = {'correct': 0, 'total': 0, 'acc': 0.0}
        continue

    # 训练子分类器（用训练集的静态特征）
    train_mask = np.isin(y_train_s_orig, group_labels)
    X_train_g = X_train_s[train_mask][:, -1, :]  # 取窗口最后一帧
    y_train_g = y_train_s_orig[train_mask]

    label_map = {old: new for new, old in enumerate(group_labels)}
    y_train_g_mapped = np.array([label_map[l] for l in y_train_g])
    y_test_g_mapped = np.array([label_map[l] for l in y_test_g])

    n_subclasses = len(group_labels)

    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_g, y_train_g_mapped, test_size=0.1, random_state=42, stratify=y_train_g_mapped
    )
    y_train_sub_oh = to_categorical(y_train_sub, n_subclasses)
    y_val_sub_oh = to_categorical(y_val_sub, n_subclasses)

    # 使用静态DNN architecture (128-64-32)
    from tensorflow.keras.models import Sequential as Seq
    model_sub = Seq([
        Input(shape=(N_FEATURES,)),
        Dense(128, kernel_regularizer=l2(1e-4)),
        BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.1),
        Dense(64, kernel_regularizer=l2(1e-4)),
        BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.1),
        Dense(32, kernel_regularizer=l2(1e-4)),
        BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.1),
        Dense(n_subclasses, activation='softmax'),
    ])
    model_sub.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # Rebuild after clear_session
    model_sub = Seq([
        Input(shape=(N_FEATURES,)),
        Dense(128, kernel_regularizer=l2(1e-4)),
        BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.1),
        Dense(64, kernel_regularizer=l2(1e-4)),
        BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.1),
        Dense(32, kernel_regularizer=l2(1e-4)),
        BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.1),
        Dense(n_subclasses, activation='softmax'),
    ])
    model_sub.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    t0 = time.time()
    model_sub.fit(
        X_train_sub, y_train_sub_oh,
        epochs=300, batch_size=128,
        validation_data=(X_val_sub, y_val_sub_oh),
        callbacks=callbacks_fn(), verbose=2
    )
    t1 = time.time()

    y_pred_sub_mapped = np.argmax(model_sub.predict(X_test_g, verbose=0), axis=1)
    y_pred_sub = np.array([group_labels[p] for p in y_pred_sub_mapped])

    correct = np.sum(y_pred_sub == y_test_g)
    acc_sub = correct / len(y_test_g) * 100

    print(f"  Train: {len(X_train_g)} | Test: {len(X_test_g)} | Correct: {correct} | Acc: {acc_sub:.2f}% | Time: {t1-t0:.1f}s")

    group_results[group_id] = {'correct': correct, 'total': len(y_test_g), 'acc': acc_sub}
    overall_correct += correct
    overall_total += len(y_test_g)

# ==================== 汇总 ====================
print("\n" + "=" * 70)
print("时序DNN分层分类结果汇总")
print("=" * 70)

print(f"\n{'实验':<35} {'准确率':>10}")
print("-" * 50)
print(f"{'CNN-1D 扁平10分类':<35} {acc_cnn10*100:>9.2f}%")
print(f"{'LSTM 扁平10分类':<35} {acc_lstm10*100:>9.2f}%")

ts_hierarchical_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0
print(f"{'CNN-1D 分层分类':<35} {ts_hierarchical_acc:>9.2f}%")

print(f"\n对比基线:")
print(f"{'  静态DNN 扁平 (最佳架构)':<35} {'~24.8%':>9}")
print(f"{'  静态DNN 分层':<35} {'56.06%':>9}")
print(f"{'  随机森林 分层':<35} {'58.87%':>9}")

print(f"\nStage 1 (CNN-1D 4-class) Accuracy: {acc_s1*100:.2f}%")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:>6.2f}% FAR={far_s1[i]*100:>6.2f}% GMA={gma_s1[i]*100:>6.2f}%")

print(f"\n各组 Stage 2 结果:")
for group_id in range(4):
    r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
    if r['total'] > 0:
        print(f"  {GROUPS[group_id]['name']:>8}: {r['correct']}/{r['total']} correct, Acc={r['acc']:.2f}%")

# 保存结果
results_path = os.path.join(output_dir, 'timeseries_dnn_results.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DNN Time-Series Classification Results\n")
    f.write(f"Window={WINDOW_SIZE}, Stride={STRIDE}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"CNN-1D Flat 10-class: {acc_cnn10*100:.2f}%\n")
    f.write(f"LSTM Flat 10-class:  {acc_lstm10*100:.2f}%\n")
    f.write(f"CNN-1D Hierarchical: {ts_hierarchical_acc:.2f}%\n\n")
    f.write(f"Stage 1 (CNN 4-class) Accuracy: {acc_s1*100:.2f}%\n")
    for i in range(4):
        f.write(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:>6.2f}% FAR={far_s1[i]*100:>6.2f}% GMA={gma_s1[i]*100:>6.2f}%\n")
    f.write(f"\nGroup Results:\n")
    for group_id in range(4):
        r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
        f.write(f"  {GROUPS[group_id]['name']}: {r['acc']:.2f}%\n")

print(f"\n结果已保存: {results_path}")
print("=" * 70)
print("时序DNN实验完成!")
print("=" * 70)
