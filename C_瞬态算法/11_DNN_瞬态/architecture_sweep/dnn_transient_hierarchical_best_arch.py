"""
DNN分层分类 — 瞬态数据 + 最佳扁平架构 A1_Vanilla_Base [64, 32]
验证：最优扁平架构在分层策略下能否超越原128-64-32的59.53%
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

data_dir = r'C:\Users\ccc\Desktop\algorithm\data'
output_dir = r'C:\Users\ccc\Desktop\algorithm\11_DNN_瞬态\architecture_sweep\output'
os.makedirs(output_dir, exist_ok=True)

KEEP_FIRST_N = 300

# 21 个瞬态 Gini 筛选特征
selected_path = r'C:\Users\ccc\Desktop\algorithm\08_特征提取_瞬态\output\selected_features.csv'
SELECTED_FEATURES = pd.read_csv(selected_path)['selected_features'].tolist()

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
# A1_Vanilla_Base 架构: [64, 32]
HIDDEN_UNITS = [64, 32]


def build_model(n_classes, name='model'):
    model = Sequential()
    model.add(Input(shape=(N_FEATURES,), name=f'{name}_input'))
    for i, units in enumerate(HIDDEN_UNITS):
        model.add(Dense(units, kernel_regularizer=l2(1e-4), name=f'{name}_d{i+1}'))
        model.add(BatchNormalization(name=f'{name}_bn{i+1}'))
        model.add(LeakyReLU(alpha=0.1, name=f'{name}_act{i+1}'))
        model.add(Dropout(0.1, name=f'{name}_do{i+1}'))
    model.add(Dense(n_classes, activation='softmax', name=f'{name}_output'))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def calculate_metrics(y_true, y_pred, n_classes):
    recall = np.zeros(n_classes)
    far = np.zeros(n_classes)
    mar = np.zeros(n_classes)
    gma = np.zeros(n_classes)
    for i in range(n_classes):
        yt = (y_true == i).astype(int)
        yp = (y_pred == i).astype(int)
        TP = np.sum((yt == 1) & (yp == 1))
        TN = np.sum((yt == 0) & (yp == 0))
        FP = np.sum((yt == 0) & (yp == 1))
        FN = np.sum((yt == 1) & (yp == 0))
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        far[i] = FP / (FP + TN) if (FP + TN) > 0 else 0
        mar[i] = FN / (TP + FN) if (TP + FN) > 0 else 0
        gma[i] = np.sqrt(recall[i] * (1 - far[i]))
    return recall, far, mar, gma


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


# ==================== 数据加载 ====================
print("=" * 70)
print("DNN 分层分类 — 瞬态数据 + 最佳架构 A1 [64, 32]")
print("=" * 70)

print(f"\n[1] 加载数据，仅保留前 {KEEP_FIRST_N} 行...")
print(f"使用特征数: {len(SELECTED_FEATURES)}")
print(f"架构: {HIDDEN_UNITS}")

all_data = []
for filename, info in FILE_CONDITIONS.items():
    df = pd.read_csv(os.path.join(data_dir, filename))
    df = df.iloc[:KEEP_FIRST_N].reset_index(drop=True)
    df['severity'] = info['severity']
    df['original_label'] = info['original_label']
    all_data.append(df)
    print(f"  {filename}: {len(df)} 行")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n总样本: {len(combined_df)} 行")

X = combined_df[SELECTED_FEATURES].values
X = np.nan_to_num(X, nan=0.0)
y_severity = combined_df['severity'].values
y_original = combined_df['original_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train_sev, y_test_sev, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y_severity, y_original, test_size=0.3, random_state=42, stratify=y_severity
)
print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

# ==================== Stage 1: 4分类 ====================
print("\n" + "=" * 70)
print("Stage 1: DNN 4-Class Severity Classification [64, 32]")
print("=" * 70)

X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(
    X_train, y_train_sev, test_size=0.1, random_state=42, stratify=y_train_sev
)
y_train_s1_oh = to_categorical(y_train_s1, 4)
y_val_s1_oh = to_categorical(y_val_s1, 4)

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model_s1 = build_model(4, 'stage1_a1')
print(f"Stage 1 参数量: {model_s1.count_params():,}")

t0 = time.time()
hist_s1 = model_s1.fit(
    X_train_s1, y_train_s1_oh,
    epochs=300, batch_size=64,
    validation_data=(X_val_s1, y_val_s1_oh),
    callbacks=callbacks_fn(), verbose=2
)
t1 = time.time()

y_pred_s1 = np.argmax(model_s1.predict(X_test, verbose=0), axis=1)
acc_s1 = accuracy_score(y_test_sev, y_pred_s1)
rec_s1, far_s1, mar_s1, gma_s1 = calculate_metrics(y_test_sev, y_pred_s1, 4)

print(f"\nStage 1 Accuracy: {acc_s1*100:.2f}% | 训练时间: {t1-t0:.1f}s")
print(f"{'Class':<20} {'Recall':>8} {'FAR':>8} {'MAR':>8} {'GMA':>8}")
print("-" * 55)
for i in range(4):
    print(f"{SEVERITY_NAMES[i]:<20} {rec_s1[i]*100:>7.2f}% {far_s1[i]*100:>7.2f}% {mar_s1[i]*100:>7.2f}% {gma_s1[i]*100:>7.2f}%")

cm_s1 = confusion_matrix(y_test_sev, y_pred_s1)
plot_cm(cm_s1, SEVERITY_NAMES,
        'Transient DNN [64,32] Stage 1: 4-Class',
        'bestarch_stage1_cm.png')

# ==================== Stage 2: 组内细分 ====================
print("\n" + "=" * 70)
print("Stage 2: DNN Intra-Group 10-Class Sub-classification [64, 32]")
print("=" * 70)

correct_mask = (y_pred_s1 == y_test_sev)
print(f"Stage 1 正确样本: {np.sum(correct_mask)} / {len(y_test_sev)}")

overall_correct = 0
overall_total = 0
group_results = {}

for group_id in range(4):
    group_labels = GROUPS[group_id]['labels']
    print(f"\n--- {GROUPS[group_id]['name']} Group (labels: {group_labels}) ---")

    if group_id == 0:
        mask = (y_test_sev == group_id)
        if np.sum(mask) > 0:
            correct = np.sum(y_pred_s1[mask] == y_test_sev[mask])
            overall_correct += correct
            overall_total += np.sum(mask)
            group_results[0] = {'correct': correct, 'total': np.sum(mask), 'acc': 100.0}
            print(f"  Normal: {np.sum(mask)} samples, all correct")
        continue

    train_mask = np.isin(y_train_orig, group_labels)
    X_train_g = X_train[train_mask]
    y_train_g = y_train_orig[train_mask]

    test_mask = (y_test_sev == group_id) & correct_mask
    X_test_g = X_test[test_mask]
    y_test_g = y_test_orig[test_mask]

    if len(X_test_g) == 0:
        group_results[group_id] = {'correct': 0, 'total': 0, 'acc': 0.0}
        continue

    label_map = {old: new for new, old in enumerate(group_labels)}
    y_train_g_mapped = np.array([label_map[l] for l in y_train_g])
    y_test_g_mapped = np.array([label_map[l] for l in y_test_g])

    n_subclasses = len(group_labels)

    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_g, y_train_g_mapped, test_size=0.1, random_state=42, stratify=y_train_g_mapped
    )
    y_train_sub_oh = to_categorical(y_train_sub, n_subclasses)
    y_val_sub_oh = to_categorical(y_val_sub, n_subclasses)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    model_sub = build_model(n_subclasses, f'stage2_a1_g{group_id}')
    t0 = time.time()
    model_sub.fit(
        X_train_sub, y_train_sub_oh,
        epochs=300, batch_size=64,
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

    group_label_names = [f'{l}%' for l in group_labels]
    cm_sub = confusion_matrix(y_test_g, y_pred_sub)
    plot_cm(cm_sub, group_label_names,
            f'Transient DNN [64,32] {GROUPS[group_id]["name"]} Group',
            f'bestarch_stage2_g{group_id}_cm.png')

# ==================== 汇总 ====================
print("\n" + "=" * 70)
print("结果汇总")
print("=" * 70)

print(f"\n{'Group':<10} {'Test':>8} {'Correct':>8} {'Acc':>10}")
print("-" * 40)
for group_id in range(4):
    r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
    if r['total'] > 0:
        print(f"{GROUPS[group_id]['name']:<10} {r['total']:>8} {r['correct']:>8} {r['acc']:>9.2f}%")

overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0
print(f"\nStage 1 (4-class) Accuracy: {acc_s1*100:.2f}%")
print(f"Overall 10-class Accuracy: {overall_acc:.2f}%")

# ==================== 对比 ====================
print("\n" + "=" * 70)
print("对比: [64,32] vs 原 [128,64,32] 瞬态分层")
print("=" * 70)
print(f"{'指标':<30} {'原 [128,64,32]':>15} {'最佳 [64,32]':>15} {'变化':>12}")
print("-" * 74)
print(f"{'Stage 1 Accuracy':<30} {'56.33%':>15} {acc_s1*100:>14.2f}% {acc_s1*100-56.33:>+11.2f}%")
print(f"{'Overall 10-class':<30} {'59.53%':>15} {overall_acc:>14.2f}% {overall_acc-59.53:>+11.2f}%")
print(f"\n{'Class':<20} {'原 [128,64,32]':>15} {'最佳 [64,32]':>15} {'变化':>12}")
print("-" * 64)
orig_recs = [99.76, 50.74, 37.41, 86.84]  # from original transient DNN
for i in range(4):
    print(f"{SEVERITY_NAMES[i]:<20} {orig_recs[i]:>14.2f}% {rec_s1[i]*100:>14.2f}% {rec_s1[i]*100-orig_recs[i]:>+11.2f}%")

# 与扁平架构扫描最佳对比
flat_best = 34.78
print(f"\n{'=' * 70}")
print(f"扁平 vs 分层 (均使用 [64, 32])")
print(f"{'=' * 70}")
print(f"  扁平 10 分类最佳: {flat_best:.2f}%")
print(f"  分层 10 分类:     {overall_acc:.2f}%")
print(f"  提升:             +{overall_acc - flat_best:.2f}%")

# 保存结果
with open(os.path.join(output_dir, 'bestarch_hierarchical_results.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DNN Hierarchical — Transient + Best Architecture [64, 32]\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Features: {len(SELECTED_FEATURES)}\n")
    f.write(f"Samples: {len(combined_df)}\n")
    f.write(f"Architecture: {HIDDEN_UNITS}\n\n")
    f.write(f"Stage 1 Accuracy: {acc_s1*100:.2f}%\n\n")
    f.write("Stage 1 Per-Class:\n")
    for i in range(4):
        f.write(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:6.2f}% FAR={far_s1[i]*100:6.2f}% MAR={mar_s1[i]*100:6.2f}% GMA={gma_s1[i]*100:6.2f}%\n")
    f.write(f"\nGroup Results:\n")
    for group_id in range(4):
        r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
        f.write(f"  {GROUPS[group_id]['name']}: {r['acc']:.2f}%\n")
    f.write(f"\nOverall 10-class Accuracy: {overall_acc:.2f}%\n")
    f.write(f"\nComparison:\n")
    f.write(f"  Original [128,64,32] Transient Hierarchical: 59.53%\n")
    f.write(f"  Best Arch [64,32] Transient Hierarchical: {overall_acc:.2f}%\n")
    f.write(f"  Best Arch [64,32] Transient Flat: 34.78%\n")

print(f"\n结果已保存至: {output_dir}/bestarch_hierarchical_results.txt")
print("=" * 70)
print("最佳架构分层分类完成!")
print("=" * 70)
