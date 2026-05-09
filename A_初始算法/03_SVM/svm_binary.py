"""
SVM二分类训练脚本 - 热泵冷却系统故障检测
正常(0%) vs 故障(5%~50%)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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

print("=" * 60)
print("SVM二分类故障检测 - 热泵冷却系统")
print("正常(0%) vs 故障(5%~50%)")
print("=" * 60)

# 1. 加载原始数据
print("\n[1] 加载原始数据...")
data_dir = r'C:\Users\ccc\Desktop\algorithm\01_源数据'

file_conditions = {
    'Heatpump_Leak_0pct.csv': 'normal',
    'Heatpump_Leak_5pct.csv': 'fault',
    'Heatpump_Leak_10pct.csv': 'fault',
    'Heatpump_Leak_20pct.csv': 'fault',
    'Heatpump_Leak_25pct.csv': 'fault',
    'Heatpump_Leak_30pct.csv': 'fault',
    'Heatpump_Leak_35pct.csv': 'fault',
    'Heatpump_Leak_40pct.csv': 'fault',
    'Heatpump_Leak_45pct.csv': 'fault',
    'Heatpump_Leak_50pct.csv': 'fault',
}

all_data = []
for filename, condition in file_conditions.items():
    filepath = f'{data_dir}/{filename}'
    df = pd.read_csv(filepath)
    df['binary_label'] = 0 if condition == 'normal' else 1
    df['condition'] = condition
    all_data.append(df)
    print(f"    {filename}: {len(df)} 行, 标签={df['binary_label'].iloc[0]} ({condition})")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n合并后总数据量: {len(combined_df)} 行")

# 2. 二分类标签分布
print("\n[2] 二分类标签分布:")
binary_dist = combined_df['binary_label'].value_counts()
print(f"    正常(0): {binary_dist[0]} 样本")
print(f"    故障(1): {binary_dist[1]} 样本")

# 3. 分离特征
feature_cols = [col for col in combined_df.columns if col not in ['#', 'time[s]', 'label', 'condition', 'binary_label']]
X = combined_df[feature_cols]
y_binary = combined_df['binary_label'].values
print(f"\n原始特征数量: {len(feature_cols)}")

# 4. 使用Gini重要性进行二分类特征筛选
print("\n[3] 基于二分类进行Gini重要性特征筛选...")
from sklearn.tree import DecisionTreeClassifier
dt_binary = DecisionTreeClassifier(random_state=42)
dt_binary.fit(X, y_binary)

gini_importance = dt_binary.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'gini_importance': gini_importance
}).sort_values('gini_importance', ascending=False)

importance_df['cumulative_importance'] = importance_df['gini_importance'].cumsum()
importance_df['rank'] = range(1, len(importance_df) + 1)
importance_df = importance_df[['rank', 'feature', 'gini_importance', 'cumulative_importance']]

print("\n    Gini重要性排名 (Top 20):")
print(importance_df.head(20).to_string(index=False))

importance_df.to_csv(f'{data_dir}/binary_gini_feature_importance.csv', index=False)

threshold = 0.95
selected_features = importance_df[importance_df['cumulative_importance'] <= threshold]['feature'].tolist()
if len(selected_features) < 10:
    selected_features = importance_df.head(10)['feature'].tolist()

print(f"\n    筛选后特征数量: {len(selected_features)}")
print(f"    选中的特征: {selected_features}")

# 5. 数据标准化
print("\n[4] 数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_df[selected_features].values)
y_binary = combined_df['binary_label'].values

# 6. 划分训练集和测试集 (7:3)
print("\n[5] 划分训练集和测试集 (7:3)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)
print(f"    训练集样本数: {len(X_train)}")
print(f"    测试集样本数: {len(X_test)}")
print(f"    训练集类别分布: 正常={np.sum(y_train==0)}, 故障={np.sum(y_train==1)}")
print(f"    测试集类别分布: 正常={np.sum(y_test==0)}, 故障={np.sum(y_test==1)}")

# 7. 过采样处理类别不平衡
print("\n[6] 过采样处理类别不平衡...")
np.random.seed(42)
X_train_resampled, y_train_resampled = simple_random_oversample(X_train, y_train)
print(f"    过采样后训练集: 正常={np.sum(y_train_resampled==0)}, 故障={np.sum(y_train_resampled==1)}")
print(f"    过采样后总样本: {len(X_train_resampled)}")

# 8. 训练SVM二分类模型
print("\n[7] 训练SVM二分类模型...")
svm_binary = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_binary.fit(X_train_resampled, y_train_resampled)
print("    训练完成!")

# 9. 模型预测
print("\n[8] 模型预测...")
y_pred = svm_binary.predict(X_test)

# 10. 计算评估指标
print("\n[9] 计算评估指标...")
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

y_true_normal = (y_test == 0).astype(int)
y_pred_normal = (y_pred == 0).astype(int)
TP_normal = np.sum((y_true_normal == 1) & (y_pred_normal == 1))
TN_normal = np.sum((y_true_normal == 0) & (y_pred_normal == 0))
FP_normal = np.sum((y_true_normal == 0) & (y_pred_normal == 1))
FN_normal = np.sum((y_true_normal == 1) & (y_pred_normal == 0))

y_true_fault = (y_test == 1).astype(int)
y_pred_fault = (y_pred == 1).astype(int)
TP_fault = np.sum((y_true_fault == 1) & (y_pred_fault == 1))
TN_fault = np.sum((y_true_fault == 0) & (y_pred_fault == 0))
FP_fault = np.sum((y_true_fault == 0) & (y_pred_fault == 1))
FN_fault = np.sum((y_true_fault == 1) & (y_pred_fault == 0))

recall_normal = TP_normal / (TP_normal + FN_normal) if (TP_normal + FN_normal) > 0 else 0
recall_fault = TP_fault / (TP_fault + FN_fault) if (TP_fault + FN_fault) > 0 else 0
far_normal = FP_normal / (FP_normal + TN_normal) if (FP_normal + TN_normal) > 0 else 0
far_fault = FP_fault / (FP_fault + TN_fault) if (FP_fault + TN_fault) > 0 else 0
mar_normal = FN_normal / (TP_normal + FN_normal) if (TP_normal + FN_normal) > 0 else 0
mar_fault = FN_fault / (TP_fault + FN_fault) if (TP_fault + FN_fault) > 0 else 0
specificity_normal = 1 - far_normal
specificity_fault = 1 - far_fault
gma_normal = np.sqrt(recall_normal * specificity_normal)
gma_fault = np.sqrt(recall_fault * specificity_fault)
avg_gma = (gma_normal + gma_fault) / 2
avg_recall = (recall_normal + recall_fault) / 2
avg_far = (far_normal + far_fault) / 2
avg_mar = (mar_normal + mar_fault) / 2

print("\n" + "=" * 60)
print("SVM二分类评估指标结果")
print("=" * 60)
print(f"\n  总体准确率 (Accuracy): {accuracy * 100:.2f}%")
print(f"\n  正常类(0)指标:")
print(f"    召回率: {recall_normal*100:.2f}%")
print(f"    误报率: {far_normal*100:.2f}%")
print(f"    漏报率: {mar_normal*100:.2f}%")
print(f"    GMA: {gma_normal*100:.2f}%")
print(f"\n  故障类(1)指标:")
print(f"    召回率: {recall_fault*100:.2f}%")
print(f"    误报率: {far_fault*100:.2f}%")
print(f"    漏报率: {mar_fault*100:.2f}%")
print(f"    GMA: {gma_fault*100:.2f}%")
print(f"\n  宏平均指标:")
print(f"    召回率: {avg_recall*100:.2f}%")
print(f"    误报率: {avg_far*100:.2f}%")
print(f"    漏报率: {avg_mar*100:.2f}%")
print(f"    GMA: {avg_gma*100:.2f}%")

print("\n[10] 分类报告...")
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Fault(1)'], digits=4))

# 11. 绘制混淆矩阵
print("\n[11] 绘制混淆矩阵...")
fig, ax = plt.subplots(figsize=(8, 6))
labels = ['Normal(0%)', 'Fault(5%~50%)']
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)
im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax, label='Proportion')
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
fmt = '.2%'
thresh = cm_normalized.max() / 2.
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, format(cm_normalized[i, j], fmt) + f'\n({cm[i,j]})',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black", fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('SVM Confusion Matrix - Binary Classification\n(Normal vs Fault)', fontsize=14)
plt.tight_layout()
output_path = f'{data_dir}/svm_binary_confusion_matrix.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"混淆矩阵已保存至: {output_path}")

# 12. 保存结果
print("\n[12] 保存结果...")
results_path = f'{data_dir}/svm_binary_evaluation_results.txt'
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("SVM二分类模型评估结果 - 热泵冷却系统故障检测\n")
    f.write("正常(0%) vs 故障(5%~50%)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"模型类型: SVM (RBF核函数)\n")
    f.write(f"特征数量: {len(selected_features)}\n")
    f.write(f"选中的特征: {selected_features}\n")
    f.write(f"训练样本数 (过采样后): {len(X_train_resampled)}\n")
    f.write(f"测试样本数: {len(X_test)}\n\n")
    f.write("-" * 70 + "\n")
    f.write("评估指标\n")
    f.write("-" * 70 + "\n")
    f.write(f"总体准确率 (Accuracy): {accuracy * 100:.2f}%\n\n")
    f.write(f"正常类(0)指标:\n")
    f.write(f"  召回率: {recall_normal*100:.2f}%\n")
    f.write(f"  误报率: {far_normal*100:.2f}%\n")
    f.write(f"  漏报率: {mar_normal*100:.2f}%\n")
    f.write(f"  GMA: {gma_normal*100:.2f}%\n\n")
    f.write(f"故障类(1)指标:\n")
    f.write(f"  召回率: {recall_fault*100:.2f}%\n")
    f.write(f"  误报率: {far_fault*100:.2f}%\n")
    f.write(f"  漏报率: {mar_fault*100:.2f}%\n")
    f.write(f"  GMA: {gma_fault*100:.2f}%\n\n")
    f.write(f"宏平均指标:\n")
    f.write(f"  召回率: {avg_recall*100:.2f}%\n")
    f.write(f"  误报率: {avg_far*100:.2f}%\n")
    f.write(f"  漏报率: {avg_mar*100:.2f}%\n")
    f.write(f"  GMA: {avg_gma*100:.2f}%\n\n")
    f.write("-" * 70 + "\n")
    f.write("分类报告\n")
    f.write("-" * 70 + "\n")
    f.write(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Fault(1)'], digits=4))

print(f"评估结果已保存至: {results_path}")

pred_results = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Correct': y_test == y_pred
})
pred_results.to_csv(f'{data_dir}/svm_binary_prediction_results.csv', index=False)
print(f"预测结果已保存至: {data_dir}/svm_binary_prediction_results.csv")

pd.DataFrame({'selected_features': selected_features}).to_csv(f'{data_dir}/svm_binary_selected_features.csv', index=False)
print(f"选中特征已保存至: {data_dir}/svm_binary_selected_features.csv")

print("\n" + "=" * 60)
print("SVM二分类训练与评估完成!")
print("=" * 60)