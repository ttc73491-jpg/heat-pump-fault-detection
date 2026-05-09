"""
随机森林分层分类 — 瞬态数据（仅保留前300行）
使用瞬态 Gini 筛选特征，复刻原始两级分层策略
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

data_dir = r'C:\Users\ccc\Desktop\algorithm\data'
output_dir = r'C:\Users\ccc\Desktop\algorithm\09_随机森林_瞬态\output'

KEEP_FIRST_N = 300

# 等待 08 脚本运行后从 selected_features.csv 读取
selected_path = r'C:\Users\ccc\Desktop\algorithm\08_特征提取_瞬态\output\selected_features.csv'
if os.path.exists(selected_path):
    SELECTED_FEATURES = pd.read_csv(selected_path)['selected_features'].tolist()
else:
    # 备用：如果还没运行特征提取
    SELECTED_FEATURES = []

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


def simple_random_oversample(X, y):
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    X_resampled, y_resampled = [], []
    for cls, count in zip(unique_classes, counts):
        cls_indices = np.where(y == cls)[0]
        X_cls = X[cls_indices]
        if count < max_count:
            n_to_add = max_count - count
            indices_to_add = np.random.choice(cls_indices, size=n_to_add, replace=True)
            X_cls = np.vstack([X_cls, X[indices_to_add]])
        X_resampled.append(X_cls)
        y_resampled.append(np.full(len(X_cls), cls))
    return np.vstack(X_resampled), np.concatenate(y_resampled)


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
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ==================== 数据加载 ====================
print("=" * 70)
print("随机森林分层分类 — 瞬态数据（仅保留前300行）")
print("=" * 70)

print(f"\n[1] 加载数据，仅保留前 {KEEP_FIRST_N} 行...")
print(f"使用特征数: {len(SELECTED_FEATURES)}")

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

X_train, X_test, y_train_sev, y_test_sev, y_train_orig, y_test_orig = train_test_split(
    X, y_severity, y_original, test_size=0.3, random_state=42, stratify=y_severity
)
print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

# ==================== Stage 1: 4分类 ====================
print("\n" + "=" * 70)
print("Stage 1: 4-Class Severity Classification")
print("=" * 70)

np.random.seed(42)
X_train_resampled, y_train_resampled = simple_random_oversample(X_train, y_train_sev)

rf_s1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_s1.fit(X_train_resampled, y_train_resampled)

y_pred_s1 = rf_s1.predict(X_test)
acc_s1 = accuracy_score(y_test_sev, y_pred_s1)
rec_s1, far_s1, mar_s1, gma_s1 = calculate_metrics(y_test_sev, y_pred_s1, 4)

print(f"\nStage 1 Accuracy: {acc_s1*100:.2f}%")
print(f"{'Class':<20} {'Recall':>8} {'FAR':>8} {'MAR':>8} {'GMA':>8}")
print("-" * 55)
for i in range(4):
    print(f"{SEVERITY_NAMES[i]:<20} {rec_s1[i]*100:>7.2f}% {far_s1[i]*100:>7.2f}% {mar_s1[i]*100:>7.2f}% {gma_s1[i]*100:>7.2f}%")

cm_s1 = confusion_matrix(y_test_sev, y_pred_s1)
plot_cm(cm_s1, SEVERITY_NAMES, 'Transient RF Stage 1: 4-Class', 'transient_rf_stage1_cm.png')

# ==================== Stage 2: 组内细分 ====================
print("\n" + "=" * 70)
print("Stage 2: Intra-Group 10-Class Sub-classification")
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
            print(f"  Normal: {np.sum(mask)} samples, all correct (no sub-classifier needed)")
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

    X_train_g_resampled, y_train_g_resampled = simple_random_oversample(X_train_g, y_train_g_mapped)

    rf_sub = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_sub.fit(X_train_g_resampled, y_train_g_resampled)

    y_pred_sub_mapped = rf_sub.predict(X_test_g)
    y_pred_sub = np.array([group_labels[p] for p in y_pred_sub_mapped])

    correct = np.sum(y_pred_sub == y_test_g)
    acc_sub = correct / len(y_test_g) * 100

    print(f"  Train: {len(X_train_g)} | Test: {len(X_test_g)} | Correct: {correct} | Acc: {acc_sub:.2f}%")

    group_results[group_id] = {'correct': correct, 'total': len(y_test_g), 'acc': acc_sub}
    overall_correct += correct
    overall_total += len(y_test_g)

    group_label_names = [f'{l}%' for l in group_labels]
    cm_sub = confusion_matrix(y_test_g, y_pred_sub)
    plot_cm(cm_sub, group_label_names,
            f'Transient RF {GROUPS[group_id]["name"]} Group',
            f'transient_rf_stage2_g{group_id}_cm.png')

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

# ==================== 三方对比 ====================
print("\n" + "=" * 70)
print("三方对比: 原始(全量) vs 稳态(去前300) vs 瞬态(仅前300)")
print("=" * 70)
print(f"{'指标':<30} {'原始(全量)':>12} {'稳态(去前300)':>14} {'瞬态(仅前300)':>14}")
print("-" * 75)
print(f"{'特征数':<30} {'20':>12} {'18':>14} {len(SELECTED_FEATURES):>14}")
print(f"{'Stage 1 Accuracy':<30} {'45.47%':>12} {'40.30%':>14} {acc_s1*100:>13.2f}%")
print(f"{'Overall 10-class':<30} {'58.87%':>12} {'55.90%':>14} {overall_acc:>13.2f}%")

print(f"\nStage 1 Per-Class Recall:")
print(f"{'Class':<20} {'原始':>8} {'稳态':>8} {'瞬态':>8}")
print("-" * 48)
orig_recs = [100.0, 28.55, 17.37, 72.32]
steady_recs = [100.0, 20.89, 8.78, 71.34]
for i in range(4):
    print(f"{SEVERITY_NAMES[i]:<20} {orig_recs[i]:>7.2f}% {steady_recs[i]:>7.2f}% {rec_s1[i]*100:>7.2f}%")

# 保存结果
with open(os.path.join(output_dir, 'transient_rf_results.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("Random Forest Hierarchical — Transient (first 300 rows only)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Features: {len(SELECTED_FEATURES)}\n")
    f.write(f"Samples: {len(combined_df)}\n\n")
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
    f.write(f"  Original (full):    58.87%\n")
    f.write(f"  Steady (remove 300): 55.90%\n")
    f.write(f"  Transient (first 300): {overall_acc:.2f}%\n")

print(f"\n结果已保存至: {output_dir}/transient_rf_results.txt")
print("=" * 70)
print("瞬态随机森林分层分类完成!")
print("=" * 70)
