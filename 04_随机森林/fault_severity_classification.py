"""
故障程度细分分类脚本 - 热泵冷却系统故障检测
方案：正常(0%) -> 轻度/中度/重度故障分组 -> 各组内10分类细分
评估指标：准确率、误报率、漏报率、几何平均准确率
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
warnings.filterwarnings('ignore')

def simple_random_oversample(X, y):
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    X_resampled = []
    y_resampled = []
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

def plot_confusion_matrix(cm, labels, title, filename, data_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Proportion')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = '.2%'
    thresh = cm_normalized.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(cm_normalized[i, j], fmt) + f'\n({cm[i,j]})',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black", fontsize=9)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{data_dir}/{filename}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  混淆矩阵已保存: {data_dir}/{filename}")

print("=" * 70)
print("故障程度细分分类 - 热泵冷却系统故障检测")
print("=" * 70)

# 1. 加载数据
print("\n[1] 加载原始数据...")
data_dir = r'C:\Users\ccc\Desktop\algorithm\01_源数据'

file_conditions = {
    'Heatpump_Leak_0pct.csv': {'severity': 0, 'original_label': 0},
    'Heatpump_Leak_5pct.csv': {'severity': 1, 'original_label': 1},
    'Heatpump_Leak_10pct.csv': {'severity': 1, 'original_label': 2},
    'Heatpump_Leak_20pct.csv': {'severity': 1, 'original_label': 3},
    'Heatpump_Leak_25pct.csv': {'severity': 2, 'original_label': 4},
    'Heatpump_Leak_30pct.csv': {'severity': 2, 'original_label': 5},
    'Heatpump_Leak_35pct.csv': {'severity': 2, 'original_label': 6},
    'Heatpump_Leak_40pct.csv': {'severity': 3, 'original_label': 7},
    'Heatpump_Leak_45pct.csv': {'severity': 3, 'original_label': 8},
    'Heatpump_Leak_50pct.csv': {'severity': 3, 'original_label': 9},
}

all_data = []
for filename, info in file_conditions.items():
    filepath = f'{data_dir}/{filename}'
    df = pd.read_csv(filepath)
    df['severity'] = info['severity']
    df['original_label'] = info['original_label']
    all_data.append(df)
    severity_names = ['正常', '轻度', '中度', '重度']
    print(f"  {filename}: {len(df)} 行, 严重度={severity_names[info['severity']]}({info['severity']})")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n合并后总数据量: {len(combined_df)} 行")

# 2. 定义特征列
feature_cols = [col for col in combined_df.columns if col not in ['#', 'time[s]', 'label', 'condition', 'severity', 'original_label']]
print(f"\n特征数量: {len(feature_cols)}")

# 3. 划分训练集和测试集 (7:3)
print("\n[2] 划分训练集和测试集 (7:3)...")
X = combined_df[feature_cols].values
y_severity = combined_df['severity'].values
y_original = combined_df['original_label'].values

X_train, X_test, y_train_sev, y_test_sev, y_train_orig, y_test_orig = train_test_split(
    X, y_severity, y_original, test_size=0.3, random_state=42, stratify=y_severity
)
print(f"  训练集样本数: {len(X_train)}")
print(f"  测试集样本数: {len(X_test)}")

# ============================================================
# 第一阶段：4分类 (正常 vs 轻度 vs 中度 vs 重度)
# ============================================================
print("\n" + "=" * 70)
print("第一阶段：4分类 (正常/轻度/中度/重度)")
print("=" * 70)

print("\n[3] 4分类过采样...")
np.random.seed(42)
X_train_resampled, y_train_resampled = simple_random_oversample(X_train, y_train_sev)
print(f"  过采样后训练集样本数: {len(X_train_resampled)}")

print("\n[4] 训练4分类随机森林模型...")
rf_4class = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_4class.fit(X_train_resampled, y_train_resampled)

y_pred_sev = rf_4class.predict(X_test)

cm_4class = confusion_matrix(y_test_sev, y_pred_sev)
acc_4class = accuracy_score(y_test_sev, y_pred_sev)
recall_4, far_4, mar_4, gma_4 = calculate_metrics(y_test_sev, y_pred_sev, 4)
avg_recall_4 = np.mean(recall_4)
avg_far_4 = np.mean(far_4)
avg_mar_4 = np.mean(mar_4)
avg_gma_4 = np.mean(gma_4)

severity_labels = ['正常(0%)', '轻度(5-20%)', '中度(25-35%)', '重度(40-50%)']

print("\n  4分类评估指标:")
print("-" * 60)
print(f"  {'类别':<15} {'召回率':<12} {'误报率':<12} {'漏报率':<12} {'GMA':<12}")
print("-" * 60)
for i in range(4):
    print(f"  {severity_labels[i]:<15} {recall_4[i]*100:>10.2f}% {far_4[i]*100:>10.2f}% {mar_4[i]*100:>10.2f}% {gma_4[i]*100:>10.2f}%")
print("-" * 60)
print(f"  {'宏平均':<15} {avg_recall_4*100:>10.2f}% {avg_far_4*100:>10.2f}% {avg_mar_4*100:>10.2f}% {avg_gma_4*100:>10.2f}%")
print("-" * 60)
print(f"\n  总体准确率: {acc_4class*100:.2f}%")

plot_confusion_matrix(cm_4class, severity_labels,
    'Confusion Matrix - 4-Class Classification\n(Normal/Light/Medium/Severe)',
    'severity_4class_confusion_matrix.png', data_dir)

# ============================================================
# 第二阶段：分组内10分类
# ============================================================
print("\n" + "=" * 70)
print("第二阶段：分组内10分类细分")
print("=" * 70)

print("\n[5] 对每个故障程度组进行组内10分类细分...")

correct_4class_mask = (y_pred_sev == y_test_sev)
print(f"  4分类正确的样本数: {np.sum(correct_4class_mask)} / {len(y_test_sev)}")

group_results = {}
overall_correct_10class = 0
overall_total_10class = 0

groups = {
    0: {'name': '正常', 'labels': [0], 'test_indices': []},
    1: {'name': '轻度', 'labels': [1, 2, 3], 'test_indices': []},
    2: {'name': '中度', 'labels': [4, 5, 6], 'test_indices': []},
    3: {'name': '重度', 'labels': [7, 8, 9], 'test_indices': []},
}

for group_id, group_info in groups.items():
    print(f"\n  --- {group_info['name']}故障组 (标签: {group_info['labels']}) ---")

    if group_id == 0:
        group_mask = (y_test_sev == group_id)
        if np.sum(group_mask) > 0:
            correct = np.sum(y_pred_sev[group_mask] == y_test_sev[group_mask])
            print(f"    正常组: {np.sum(group_mask)} 样本, 全部正确分类 (无需细分)")
            group_results[group_id] = {'correct': np.sum(group_mask), 'total': np.sum(group_mask), 'accuracy': 100.0}
            overall_correct_10class += np.sum(group_mask)
            overall_total_10class += np.sum(group_mask)
        continue

    group_labels = group_info['labels']
    train_mask = np.isin(y_train_orig, group_labels)
    X_train_group = X_train[train_mask]
    y_train_group = y_train_orig[train_mask]

    group_test_mask = (y_test_sev == group_id) & correct_4class_mask
    X_test_group = X_test[group_test_mask]
    y_test_group = y_test_orig[group_test_mask]

    if len(X_test_group) == 0:
        print(f"    无测试样本")
        group_results[group_id] = {'correct': 0, 'total': 0, 'accuracy': 0.0}
        continue

    label_mapping = {old_label: new_label for new_label, old_label in enumerate(group_labels)}
    y_train_group_mapped = np.array([label_mapping[l] for l in y_train_group])
    y_test_group_mapped = np.array([label_mapping[l] for l in y_test_group])

    X_train_group_resampled, y_train_group_resampled = simple_random_oversample(X_train_group, y_train_group_mapped)

    rf_group = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_group.fit(X_train_group_resampled, y_train_group_resampled)

    y_pred_group_mapped = rf_group.predict(X_test_group)
    y_pred_group = np.array([group_labels[p] for p in y_pred_group_mapped])

    correct = np.sum(y_pred_group == y_test_group)
    group_acc = correct / len(y_test_group) * 100

    print(f"    训练样本: {len(X_train_group)}, 过采样后: {len(X_train_group_resampled)}")
    print(f"    测试样本: {len(X_test_group)}, 正确: {correct}, 准确率: {group_acc:.2f}%")

    group_results[group_id] = {'correct': correct, 'total': len(X_test_group), 'accuracy': group_acc}
    overall_correct_10class += correct
    overall_total_10class += len(X_test_group)

    cm_group = confusion_matrix(y_test_group, y_pred_group)
    group_labels_names = [f'{l}%' for l in group_labels]
    plot_confusion_matrix(cm_group, group_labels_names,
        f'Confusion Matrix - {group_info["name"]} Group\n(10-class Sub-classification)',
        f'severity_{group_id}_subclass_confusion_matrix.png', data_dir)

print("\n" + "=" * 70)
print("分组内10分类结果汇总")
print("=" * 70)

print("\n各组详细结果:")
print("-" * 50)
for group_id, group_info in groups.items():
    result = group_results.get(group_id, {'correct': 0, 'total': 0, 'accuracy': 0})
    if result['total'] > 0:
        print(f"  {group_info['name']}组: {result['correct']}/{result['total']} 正确, 准确率: {result['accuracy']:.2f}%")
    else:
        print(f"  {group_info['name']}组: 无测试样本")
print("-" * 50)

overall_10class_accuracy = overall_correct_10class / overall_total_10class * 100 if overall_total_10class > 0 else 0
print(f"\n  分组内10分类总体准确率: {overall_10class_accuracy:.2f}%")
print(f"  (基于 {overall_total_10class} 个测试样本)")

print("\n" + "=" * 70)
print("与原始10分类对比")
print("=" * 70)

print(f"\n  4分类 (正常/轻度/中度/重度) 准确率: {acc_4class*100:.2f}%")
print(f"  分组内10分类总体准确率: {overall_10class_accuracy:.2f}%")

comparison_results = {
    '分类方法': ['4分类', '分组内10分类 (整体)'],
    '准确率': [f'{acc_4class*100:.2f}%', f'{overall_10class_accuracy:.2f}%']
}
comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(f'{data_dir}/classification_comparison.csv', index=False)
print(f"\n对比结果已保存: {data_dir}/classification_comparison.csv")

print("\n" + "=" * 70)
print("故障程度细分分类完成!")
print("=" * 70)