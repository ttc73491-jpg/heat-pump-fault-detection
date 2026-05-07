"""
SVM训练脚本 - 热泵冷却系统故障检测
使用GridSearchCV进行网格搜索最优C和gamma参数
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
import time
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
print("SVM超参数调优 - 热泵冷却系统故障检测")
print("=" * 60)

# 1. 加载数据
print("\n[1] 加载预处理后的数据...")
data_dir = r'C:\Users\ccc\Desktop\algorithm\01_源数据'
df = pd.read_csv(f'{data_dir}/processed_data_with_selected_features.csv')

feature_cols = [col for col in df.columns if col not in ['label', 'condition']]
X = df[feature_cols].values
y = df['label'].values

print(f"    样本数量: {len(X)}")
print(f"    特征数量: {len(feature_cols)}")
print(f"    类别数量: {len(np.unique(y))}")

# 2. 数据标准化
print("\n[2] 数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集 (7:3)
print("\n[3] 划分训练集和测试集 (7:3)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"    训练集样本数: {len(X_train)}")
print(f"    测试集样本数: {len(X_test)}")

# 4. 使用随机过采样处理类别不平衡
print("\n[4] 使用随机过采样处理类别不平衡...")
np.random.seed(42)
X_train_resampled, y_train_resampled = simple_random_oversample(X_train, y_train)
print(f"    过采样后训练集样本数: {len(X_train_resampled)}")

# 5. 定义超参数网格
print("\n[5] 定义超参数搜索网格...")
param_grid = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
print(f"    C候选值: {param_grid['C']}")
print(f"    gamma候选值: {param_grid['gamma']}")
print(f"    共 {len(param_grid['C']) * len(param_grid['gamma'])} 种组合")

# 6. 网格搜索
print("\n[6] 开始网格搜索...")
start_time = time.time()

svm_model = SVC(kernel='rbf', random_state=42)
grid_search = GridSearchCV(
    svm_model, param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_resampled, y_train_resampled)

elapsed_time = time.time() - start_time
print(f"\n    网格搜索完成! 耗时: {elapsed_time:.1f}秒")

print(f"\n    最佳参数: C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}")
print(f"    交叉验证最佳得分: {grid_search.best_score_*100:.2f}%")

# 7. 使用最佳参数训练最终模型
print("\n[7] 使用最佳参数训练最终模型...")
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_resampled, y_train_resampled)

# 8. 模型预测
print("\n[8] 模型预测...")
y_pred = best_svm.predict(X_test)

# 9. 计算评估指标
print("\n[9] 计算评估指标...")
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

n_classes = len(np.unique(y))
recall_per_class = np.zeros(n_classes)
far_per_class = np.zeros(n_classes)
mar_per_class = np.zeros(n_classes)

for i in range(n_classes):
    y_true_binary = (y_test == i).astype(int)
    y_pred_binary = (y_pred == i).astype(int)
    TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    recall_per_class[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
    far_per_class[i] = FP / (FP + TN) if (FP + TN) > 0 else 0
    mar_per_class[i] = FN / (TP + FN) if (TP + FN) > 0 else 0

specificity_per_class = 1 - far_per_class
gma_per_class = np.sqrt(recall_per_class * specificity_per_class)

avg_recall = np.mean(recall_per_class)
avg_far = np.mean(far_per_class)
avg_mar = np.mean(mar_per_class)
avg_gma = np.mean(gma_per_class)

print("\n" + "=" * 60)
print("SVM超参数调优 - 评估指标结果")
print("=" * 60)
print(f"\n  最佳参数: C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}")
print(f"  总体准确率 (Accuracy): {accuracy * 100:.2f}%")
print(f"\n  各类别指标:")
print("-" * 70)
print(f"  {'类别':<8} {'召回率':<12} {'误报率':<12} {'漏报率':<12} {'GMA':<12}")
print("-" * 70)
for i in range(n_classes):
    print(f"  {i:<8} {recall_per_class[i]*100:>10.2f}% {far_per_class[i]*100:>10.2f}% {mar_per_class[i]*100:>10.2f}% {gma_per_class[i]*100:>10.2f}%")
print("-" * 70)
print(f"  {'宏平均':<8} {avg_recall*100:>10.2f}% {avg_far*100:>10.2f}% {avg_mar*100:>10.2f}% {avg_gma*100:>10.2f}%")
print("-" * 70)

print("\n[10] 分类报告...")
target_names = [f'Class {i}' for i in range(n_classes)]
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# 10. 绘制混淆矩阵
print("\n[11] 绘制混淆矩阵...")
fig, ax = plt.subplots(figsize=(12, 10))
labels = ['0% (Normal)', '5%', '10%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']
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
        ax.text(j, i, format(cm_normalized[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black", fontsize=9)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Confusion Matrix - SVM with RBF Kernel (Tuned)\nBest Params: C={grid_search.best_params_["C"]}, gamma={grid_search.best_params_["gamma"]}', fontsize=14)
plt.tight_layout()
output_path = f'{data_dir}/svm_tuned_confusion_matrix.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"混淆矩阵已保存至: {output_path}")

# 11. 保存结果
print("\n[12] 保存评估结果...")
results_path = f'{data_dir}/svm_tuned_evaluation_results.txt'
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("SVM模型超参数调优评估结果 - 热泵冷却系统故障检测\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"模型类型: SVM (RBF核函数)\n")
    f.write(f"最佳参数: C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}\n")
    f.write(f"特征数量: {len(feature_cols)}\n")
    f.write(f"训练样本数 (过采样后): {len(X_train_resampled)}\n")
    f.write(f"测试样本数: {len(X_test)}\n\n")
    f.write("-" * 70 + "\n")
    f.write("评估指标\n")
    f.write("-" * 70 + "\n")
    f.write(f"总体准确率 (Accuracy): {accuracy * 100:.2f}%\n")
    f.write(f"宏平均召回率 (Macro Recall): {avg_recall * 100:.2f}%\n")
    f.write(f"宏平均误报率 (Macro FAR): {avg_far * 100:.2f}%\n")
    f.write(f"宏平均漏报率 (Macro MAR): {avg_mar * 100:.2f}%\n")
    f.write(f"几何平均准确率 (GMA): {avg_gma * 100:.2f}%\n\n")
    f.write("-" * 70 + "\n")
    f.write("各类别详细指标\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'类别':<12} {'召回率':<12} {'误报率':<12} {'漏报率':<12} {'GMA':<12}\n")
    for i in range(n_classes):
        f.write(f"Class {i:<6} {recall_per_class[i]*100:>10.2f}% {far_per_class[i]*100:>10.2f}% {mar_per_class[i]*100:>10.2f}% {gma_per_class[i]*100:>10.2f}%\n")
    f.write("\n" + "-" * 70 + "\n")
    f.write("分类报告\n")
    f.write("-" * 70 + "\n")
    f.write(classification_report(y_test, y_pred, target_names=target_names, digits=4))

print(f"评估结果已保存至: {results_path}")

pred_results = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Correct': y_test == y_pred
})
pred_results.to_csv(f'{data_dir}/svm_tuned_prediction_results.csv', index=False)
print(f"预测结果已保存至: {data_dir}/svm_tuned_prediction_results.csv")

best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv(f'{data_dir}/svm_best_params.csv', index=False)
print(f"最佳参数已保存至: {data_dir}/svm_best_params.csv")

print("\n" + "=" * 60)
print("SVM超参数调优完成!")
print("=" * 60)