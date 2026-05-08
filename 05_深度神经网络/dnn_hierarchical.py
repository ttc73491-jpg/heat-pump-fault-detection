"""
DNN分层分类 — 复刻RF/SVM两级策略
Stage 1: 4分类 (正常/轻度/中度/重度)
Stage 2: 组内3分类细分
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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

N_FEATURES = len(SELECTED_FEATURES)


def build_model(n_classes, name='model'):
    """构建标准DNN: 128-64-32"""
    model = Sequential([
        Input(shape=(N_FEATURES,), name=f'{name}_input'),
        Dense(128, kernel_regularizer=l2(1e-4), name=f'{name}_d1'),
        BatchNormalization(name=f'{name}_bn1'),
        LeakyReLU(alpha=0.1, name=f'{name}_act1'),
        Dropout(0.1, name=f'{name}_do1'),
        Dense(64, kernel_regularizer=l2(1e-4), name=f'{name}_d2'),
        BatchNormalization(name=f'{name}_bn2'),
        LeakyReLU(alpha=0.1, name=f'{name}_act2'),
        Dropout(0.1, name=f'{name}_do2'),
        Dense(32, kernel_regularizer=l2(1e-4), name=f'{name}_d3'),
        BatchNormalization(name=f'{name}_bn3'),
        LeakyReLU(alpha=0.1, name=f'{name}_act3'),
        Dropout(0.1, name=f'{name}_do3'),
        Dense(n_classes, activation='softmax', name=f'{name}_output'),
    ], name=name)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def calculate_metrics(y_true, y_pred, n_classes):
    recall_per_class = np.zeros(n_classes)
    far_per_class = np.zeros(n_classes)
    mar_per_class = np.zeros(n_classes)
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
        mar_per_class[i] = FN / (TP + FN) if (TP + FN) > 0 else 0
        specificity = 1 - far_per_class[i]
        gma_per_class[i] = np.sqrt(recall_per_class[i] * specificity)
    return recall_per_class, far_per_class, mar_per_class, gma_per_class


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


# ==================== 数据加载 ====================
print("=" * 70)
print("DNN 分层分类 — 两级策略")
print("=" * 70)

all_data = []
for filename, info in FILE_CONDITIONS.items():
    df = pd.read_csv(os.path.join(data_dir, filename))
    df['severity'] = info['severity']
    df['original_label'] = info['original_label']
    all_data.append(df)
combined_df = pd.concat(all_data, ignore_index=True)
print(f"总样本: {len(combined_df)}")

X = combined_df[SELECTED_FEATURES].values
X = np.nan_to_num(X, nan=0.0)
y_severity = combined_df['severity'].values
y_original = combined_df['original_label'].values

# StandardScaler + 7:3 split (stratify by severity, same as RF)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train_sev, y_test_sev, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y_severity, y_original, test_size=0.3, random_state=42, stratify=y_severity
)
print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

callbacks_fn = lambda: [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0),
]

# ==================== Stage 1: 4分类 ====================
print("\n" + "=" * 70)
print("Stage 1: 4-Class Severity Classification")
print("=" * 70)

# Split validation from training (no oversampling)
X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(
    X_train, y_train_sev, test_size=0.1, random_state=42, stratify=y_train_sev
)
y_train_s1_oh = to_categorical(y_train_s1, 4)
y_val_s1_oh = to_categorical(y_val_s1, 4)

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model_s1 = build_model(4, 'stage1')
print(f"Stage 1 参数量: {model_s1.count_params():,}")

t0 = time.time()
hist_s1 = model_s1.fit(
    X_train_s1, y_train_s1_oh,
    epochs=300, batch_size=128,
    validation_data=(X_val_s1, y_val_s1_oh),
    callbacks=callbacks_fn(), verbose=2
)
t1 = time.time()

y_pred_s1 = np.argmax(model_s1.predict(X_test, verbose=0), axis=1)
acc_s1 = accuracy_score(y_test_sev, y_pred_s1)
rec_s1, far_s1, mar_s1, gma_s1 = calculate_metrics(y_test_sev, y_pred_s1, 4)

print(f"\nStage 1 准确率: {acc_s1*100:.2f}% | 训练时间: {t1-t0:.1f}s")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:>6.2f}% FAR={far_s1[i]*100:>6.2f}% GMA={gma_s1[i]*100:>6.2f}%")

plot_cm(confusion_matrix(y_test_sev, y_pred_s1), SEVERITY_NAMES,
        'Stage 1: 4-Class Severity', 'hierarchical_stage1_cm.png')

# ==================== Stage 2: 组内细分 ====================
print("\n" + "=" * 70)
print("Stage 2: Intra-Group 10-Class Sub-classification")
print("=" * 70)

# Only samples where Stage 1 prediction is CORRECT
correct_mask = (y_pred_s1 == y_test_sev)
print(f"Stage 1 正确样本: {np.sum(correct_mask)} / {len(y_test_sev)}")

overall_correct = 0
overall_total = 0
group_results = {}

for group_id in range(4):
    group_info = GROUPS[group_id]
    group_labels = group_info['labels']
    print(f"\n--- {group_info['name']} Group (labels: {group_labels}) ---")

    if group_id == 0:
        # Normal — 直接正确（只有1个类别）
        mask = (y_test_sev == group_id)
        if np.sum(mask) > 0:
            correct = np.sum(y_pred_s1[mask] == y_test_sev[mask])
            overall_correct += correct
            overall_total += correct
            print(f"  Normal: {np.sum(mask)} samples, all correct (no sub-classifier needed)")
            group_results[0] = {'correct': correct, 'total': np.sum(mask), 'acc': 100.0}
        continue

    # 训练子分类器
    train_mask = np.isin(y_train_orig, group_labels)
    X_train_g = X_train[train_mask]
    y_train_g = y_train_orig[train_mask]

    # 只对Stage 1正确的测试样本做Stage 2
    test_mask = (y_test_sev == group_id) & correct_mask
    X_test_g = X_test[test_mask]
    y_test_g = y_test_orig[test_mask]

    if len(X_test_g) == 0:
        group_results[group_id] = {'correct': 0, 'total': 0, 'acc': 0.0}
        continue

    # 标签重映射 (e.g. [1,2,3] -> [0,1,2])
    label_map = {old: new for new, old in enumerate(group_labels)}
    y_train_g_mapped = np.array([label_map[l] for l in y_train_g])
    y_test_g_mapped = np.array([label_map[l] for l in y_test_g])

    n_subclasses = len(group_labels)

    # Split validation
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_g, y_train_g_mapped, test_size=0.1, random_state=42, stratify=y_train_g_mapped
    )
    y_train_sub_oh = to_categorical(y_train_sub, n_subclasses)
    y_val_sub_oh = to_categorical(y_val_sub, n_subclasses)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    model_sub = build_model(n_subclasses, f'stage2_g{group_id}')
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

    group_results[group_id] = {'correct': correct, 'total': len(X_test_g), 'acc': acc_sub}
    overall_correct += correct
    overall_total += len(X_test_g)

    # 混淆矩阵
    group_label_names = [f'{l}%' for l in group_labels]
    cm_sub = confusion_matrix(y_test_g, y_pred_sub)
    plot_cm(cm_sub, group_label_names,
            f'{group_info["name"]} Group (labels: {group_labels})',
            f'hierarchical_stage2_g{group_id}_cm.png')

# ==================== 汇总 ====================
print("\n" + "=" * 70)
print("分层分类结果汇总")
print("=" * 70)

print("\n各组详细结果:")
for group_id in range(4):
    r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
    if r['total'] > 0:
        print(f"  {GROUPS[group_id]['name']:>8}: {r['correct']}/{r['total']} correct, Acc={r['acc']:.2f}%")

overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0
print(f"\nStage 1 (4-class) Accuracy: {acc_s1*100:.2f}%")
print(f"Overall 10-class Accuracy: {overall_acc:.2f}%")
print(f"(Based on {overall_total} test samples correctly routed through Stage 1)")

# 保存汇总
with open(os.path.join(output_dir, 'hierarchical_dnn_results.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DNN Hierarchical Classification Results\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Stage 1 (4-class) Accuracy: {acc_s1*100:.2f}%\n\n")
    f.write("Stage 1 Per-Class Metrics:\n")
    f.write(f"{'Class':<20} {'Recall':>10} {'FAR':>10} {'GMA':>10}\n")
    for i in range(4):
        f.write(f"{SEVERITY_NAMES[i]:<20} {rec_s1[i]*100:>9.2f}% {far_s1[i]*100:>9.2f}% {gma_s1[i]*100:>9.2f}%\n")
    f.write(f"\nGroup Results:\n")
    for group_id in range(4):
        r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
        f.write(f"  {GROUPS[group_id]['name']}: {r['acc']:.2f}%\n")
    f.write(f"\nOverall 10-class Accuracy: {overall_acc:.2f}%\n")

print(f"\n结果已保存: {os.path.join(output_dir, 'hierarchical_dnn_results.txt')}")
print("=" * 70)
print("DNN分层分类完成!")
print("=" * 70)
