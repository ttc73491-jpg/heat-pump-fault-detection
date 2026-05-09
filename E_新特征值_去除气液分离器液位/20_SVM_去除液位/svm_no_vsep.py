"""
SVM — 去除V_sep_liq (38特征数据集)
扁平10分类 + 分层分类（4-class Stage 1 + 组内细分）
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(SCRIPT_DIR, '..', '17_新数据_去除液位')
output_dir = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(output_dir, exist_ok=True)

selected_path = os.path.join(SCRIPT_DIR, '..', '18_新特征值去除液位_特征提取', 'output', 'selected_features.csv')
SELECTED_FEATURES = pd.read_csv(selected_path)['selected_features'].tolist()

TARGET_NAMES = ['0%', '5%', '10%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']

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
print(f"SVM — 去除V_sep_liq 38特征数据集 ({len(SELECTED_FEATURES)} selected features)")
print("=" * 70)

print(f"\n[1] 加载数据...")
print(f"使用特征数: {len(SELECTED_FEATURES)}")

all_data = []
for filename, info in FILE_CONDITIONS.items():
    df = pd.read_csv(os.path.join(data_dir, filename))
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

# ==================== 扁平10分类 ====================
print("\n" + "=" * 70)
print("[2] SVM 扁平 10-Class Classification")
print("=" * 70)

X_train_f, X_test_f, y_train_10, y_test_10 = train_test_split(
    X_scaled, y_original, test_size=0.3, random_state=42, stratify=y_original
)

np.random.seed(42)
X_train_resampled_10, y_train_resampled_10 = simple_random_oversample(X_train_f, y_train_10)

svm_flat = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_flat.fit(X_train_resampled_10, y_train_resampled_10)

y_pred_flat = svm_flat.predict(X_test_f)
acc_flat = accuracy_score(y_test_10, y_pred_flat)
rec_flat, far_flat, mar_flat, gma_flat = calculate_metrics(y_test_10, y_pred_flat, 10)

print(f"\n扁平10分类准确率: {acc_flat*100:.2f}%")
print(f"\n{'类别':<8} {'Recall':>8} {'FAR':>8} {'MAR':>8} {'GMA':>8}")
print("-" * 42)
for i in range(10):
    print(f"{TARGET_NAMES[i]:<8} {rec_flat[i]*100:>7.2f}% {far_flat[i]*100:>7.2f}% {mar_flat[i]*100:>7.2f}% {gma_flat[i]*100:>7.2f}%")

cm_flat = confusion_matrix(y_test_10, y_pred_flat)
plot_cm(cm_flat, TARGET_NAMES, f'SVM Flat 10-Class ({len(SELECTED_FEATURES)} features, No V_sep_liq)\nAccuracy: {acc_flat*100:.2f}%', 'svm_flat_10class_cm.png')

# ==================== Stage 1: 4分类 ====================
print("\n" + "=" * 70)
print("[3] SVM Stage 1: 4-Class Severity Classification")
print("=" * 70)

np.random.seed(42)
X_train_resampled, y_train_resampled = simple_random_oversample(X_train, y_train_sev)

svm_s1 = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_s1.fit(X_train_resampled, y_train_resampled)

y_pred_s1 = svm_s1.predict(X_test)
acc_s1 = accuracy_score(y_test_sev, y_pred_s1)
rec_s1, far_s1, mar_s1, gma_s1 = calculate_metrics(y_test_sev, y_pred_s1, 4)

print(f"\nStage 1 Accuracy: {acc_s1*100:.2f}%")
print(f"{'Class':<20} {'Recall':>8} {'FAR':>8} {'MAR':>8} {'GMA':>8}")
print("-" * 55)
for i in range(4):
    print(f"{SEVERITY_NAMES[i]:<20} {rec_s1[i]*100:>7.2f}% {far_s1[i]*100:>7.2f}% {mar_s1[i]*100:>7.2f}% {gma_s1[i]*100:>7.2f}%")

cm_s1 = confusion_matrix(y_test_sev, y_pred_s1)
plot_cm(cm_s1, SEVERITY_NAMES, f'SVM Stage 1: 4-Class Severity ({len(SELECTED_FEATURES)} features, No V_sep_liq)', 'svm_stage1_cm.png')

# ==================== Stage 2: 组内细分 ====================
print("\n" + "=" * 70)
print("[4] SVM Stage 2: Intra-Group Sub-classification")
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

    X_train_g_resampled, y_train_g_resampled = simple_random_oversample(X_train_g, y_train_g_mapped)

    svm_sub = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_sub.fit(X_train_g_resampled, y_train_g_resampled)

    y_pred_sub_mapped = svm_sub.predict(X_test_g)
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
            f'SVM {GROUPS[group_id]["name"]} Group ({len(SELECTED_FEATURES)} features, No V_sep_liq)',
            f'svm_stage2_g{group_id}_cm.png')

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
print(f"\n扁平 10 分类 Accuracy: {acc_flat*100:.2f}%")
print(f"Stage 1 (4-class) Accuracy: {acc_s1*100:.2f}%")
print(f"分层 Overall 10-class Accuracy: {overall_acc:.2f}%")

# ==================== 对比 ====================
print("\n" + "=" * 70)
print("对比: 去V_sep_liq(38→27) vs 新特征(39→15) vs 原始(30→20)")
print("=" * 70)
print(f"{'指标':<30} {'去液位SVM':>12} {'新特征SVM':>12} {'原始SVM':>12}")
print("-" * 68)
print(f"{'特征数':<30} {len(SELECTED_FEATURES):>12} {'15':>12} {'20':>12}")
print(f"{'Stage 1 Accuracy':<30} {acc_s1*100:>11.2f}% {'99.74%':>12} {'43.92%':>12}")
print(f"{'Overall 10-class':<30} {overall_acc:>11.2f}% {'99.28%':>12} {'53.25%':>12}")

with open(os.path.join(output_dir, 'svm_no_vsep_results.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write(f"SVM — 38 Features No V_sep_liq ({len(SELECTED_FEATURES)} selected)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Features: {len(SELECTED_FEATURES)}\n")
    f.write(f"Features list: {SELECTED_FEATURES}\n")
    f.write(f"Samples: {len(combined_df)}\n\n")
    f.write(f"Flat 10-Class Accuracy: {acc_flat*100:.2f}%\n\n")
    f.write("Flat 10-Class Per-Class:\n")
    for i in range(10):
        f.write(f"  {TARGET_NAMES[i]:<8} Recall={rec_flat[i]*100:6.2f}% FAR={far_flat[i]*100:6.2f}% MAR={mar_flat[i]*100:6.2f}% GMA={gma_flat[i]*100:6.2f}%\n")
    f.write(f"\nStage 1 Accuracy: {acc_s1*100:.2f}%\n\n")
    f.write("Stage 1 Per-Class:\n")
    for i in range(4):
        f.write(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:6.2f}% FAR={far_s1[i]*100:6.2f}% MAR={mar_s1[i]*100:6.2f}% GMA={gma_s1[i]*100:6.2f}%\n")
    f.write(f"\nGroup Results:\n")
    for group_id in range(4):
        r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
        f.write(f"  {GROUPS[group_id]['name']}: {r['acc']:.2f}%\n")
    f.write(f"\nOverall 10-class Accuracy (Hierarchical): {overall_acc:.2f}%\n")
    f.write(f"\nComparison:\n")
    f.write(f"  New Features SVM (39→15):  99.28%\n")
    f.write(f"  Original SVM (30→20):      53.25%\n")
    f.write(f"  No V_sep_liq SVM (38→{len(SELECTED_FEATURES)}):  {overall_acc:.2f}%\n")

print(f"\n结果已保存至: {output_dir}/svm_no_vsep_results.txt")
print("=" * 70)
print("SVM去除液位分类完成!")
print("=" * 70)
