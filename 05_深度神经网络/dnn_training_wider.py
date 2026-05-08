"""
深度神经网络(DNN)训练脚本 - 热泵冷却系统故障检测
参考论文：基于深度神经网络的互联网数据中心冷却系统数据驱动故障检测
网络结构: 12（输入） → 64 → 32 → 16 → 10（输出）

参数USE_OVERSAMPLE控制是否使用过采样：
- False: 无过采样版本
- True: 有过采样版本
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(42)
tf.random.set_seed(42)

# ========== 配置 ==========
USE_OVERSAMPLE = True  # True=有过采样, False=无过采样
SUFFIX = "_no_oversample" if not USE_OVERSAMPLE else "_with_oversample"
# ========================

# 新网络结构配置
NETWORK_STRUCTURE = "12-64-32-16-10"

print("=" * 70)
print("深度神经网络训练 - 热泵冷却系统故障检测")
print(f"网络结构: {NETWORK_STRUCTURE}")
print(f"过采样: {'是' if USE_OVERSAMPLE else '否'}")
print("=" * 70)

# 1. 数据路径和特征配置
data_dir = r'C:\Users\ccc\Desktop\algorithm\01_源数据'
output_dir = r'C:\Users\ccc\Desktop\algorithm\05_深度神经网络\dnn_wider_network'

# Gini重要性排名前12的特征（累计75.03%）
selected_features = [
    'h_gc_mid[kJ/kg]',
    'P_dis[bar]',
    'h_dis[kJ/kg]',
    'P_gc_out[bar]',
    'Q_heat_s2[kW]',
    'Q_heat_s1[kW]',
    'T_mid[degC]',
    'h_eva_in[kJ/kg]',
    'T_air_in[degC]',
    'P_gc_mid[bar]',
    'W_comp[kW]',
    'T_eva_out[degC]'
]

# 工况文件及标签映射
file_label_map = {
    'Heatpump_Leak_0pct.csv': 0,
    'Heatpump_Leak_5pct.csv': 1,
    'Heatpump_Leak_10pct.csv': 2,
    'Heatpump_Leak_20pct.csv': 3,
    'Heatpump_Leak_25pct.csv': 4,
    'Heatpump_Leak_30pct.csv': 5,
    'Heatpump_Leak_35pct.csv': 6,
    'Heatpump_Leak_40pct.csv': 7,
    'Heatpump_Leak_45pct.csv': 8,
    'Heatpump_Leak_50pct.csv': 9,
}

# 2. 读取数据
print("\n[1] 读取数据...")
all_data = []
for filename, label in file_label_map.items():
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    df['label'] = label
    all_data.append(df)
    print(f"  {filename}: {len(df)} 行, 标签={label}")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n合并后总数据量: {len(combined_df)} 行")

# 3. 提取特征和标签
X = combined_df[selected_features].values
y = combined_df['label'].values
X = np.nan_to_num(X, nan=0.0)

print(f"特征数量: {len(selected_features)}")
print(f"样本数量: {len(X)}")
print(f"类别数量: {len(np.unique(y))}")

# 4. 数据标准化
print("\n[2] 数据标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 划分训练集和测试集 (7:3)
print("\n[3] 划分训练集和测试集 (7:3)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  训练集样本数: {len(X_train)}")
print(f"  测试集样本数: {len(X_test)}")

# 5.1 随机过采样（可选）
if USE_OVERSAMPLE:
    print("\n[4] 随机过采样处理类别不平衡...")
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

    np.random.seed(42)
    X_train, y_train = simple_random_oversample(X_train, y_train)
    print(f"  过采样后训练集样本数: {len(X_train)}")
else:
    print("\n[4] 不使用过采样")

# 6. 标签One-Hot编码
n_classes = len(np.unique(y))
y_train_onehot = to_categorical(y_train, num_classes=n_classes)
y_test_onehot = to_categorical(y_test, num_classes=n_classes)

# 7. 构建DNN模型 (12 → 64 → 32 → 16 → 10)
print("\n[5] 构建DNN模型 (12 → 64 → 32 → 16 → 10)...")
model = Sequential([
    Input(shape=(len(selected_features),)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 8. 训练模型
print("\n[6] 训练模型 (epochs=400, batch_size=64)...")
history = model.fit(
    X_train, y_train_onehot,
    epochs=400,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# 9. 模型预测
print("\n[7] 模型预测...")
y_pred_onehot = model.predict(X_test)
y_pred = np.argmax(y_pred_onehot, axis=1)

# 10. 计算评估指标
print("\n[8] 计算评估指标...")
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

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

print("\n" + "=" * 70)
print("DNN模型评估指标结果")
print("=" * 70)
print(f"\n  模型配置: {NETWORK_STRUCTURE} (64-32-16, Dropout 0.2)")
print(f"  过采样: {'是' if USE_OVERSAMPLE else '否'}")
print(f"  特征数量: {len(selected_features)}")
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

print("\n[9] 分类报告...")
target_names = [f'{i}%' for i in range(10)]
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# 11. 绘制混淆矩阵
print("\n[10] 绘制混淆矩阵...")
fig, ax = plt.subplots(figsize=(12, 10))
labels = ['0%', '5%', '10%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']
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
ax.set_title(f'DNN Confusion Matrix ({NETWORK_STRUCTURE})\nOversample: {"Yes" if USE_OVERSAMPLE else "No"}', fontsize=14)
plt.tight_layout()
output_path = os.path.join(output_dir, f'dnn_confusion_matrix{SUFFIX}.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"混淆矩阵已保存至: {output_path}")

# 12. 绘制训练历史
print("\n[11] 绘制训练历史...")
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('DNN Training - Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('DNN Training - Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
history_path = os.path.join(output_dir, f'dnn_training_history{SUFFIX}.png')
plt.savefig(history_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"训练历史已保存至: {history_path}")

# 13. 保存结果
print("\n[12] 保存评估结果...")
results_path = os.path.join(output_dir, f'dnn_evaluation_results{SUFFIX}.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("DNN模型评估结果 - 热泵冷却系统故障检测\n")
    f.write(f"网络结构: {NETWORK_STRUCTURE}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"模型配置: {NETWORK_STRUCTURE} (64-32-16, Dropout 0.2)\n")
    f.write(f"过采样: {'是' if USE_OVERSAMPLE else '否'}\n")
    f.write(f"激活函数: ReLU\n")
    f.write(f"优化器: Adam (learning_rate=0.01)\n")
    f.write(f"训练轮数: 400\n")
    f.write(f"批次大小: 64\n")
    f.write(f"特征数量: {len(selected_features)}\n")
    f.write(f"选中的特征: {selected_features}\n")
    f.write(f"训练样本数: {len(X_train)}\n")
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
        f.write(f"{i:<12} {recall_per_class[i]*100:>10.2f}% {far_per_class[i]*100:>10.2f}% {mar_per_class[i]*100:>10.2f}% {gma_per_class[i]*100:>10.2f}%\n")
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
pred_results.to_csv(os.path.join(output_dir, f'dnn_prediction_results{SUFFIX}.csv'), index=False)
print(f"预测结果已保存至: {output_dir}/dnn_prediction_results{SUFFIX}.csv")

model.save(os.path.join(output_dir, f'dnn_model{SUFFIX}.keras'))
print(f"模型已保存至: {output_dir}/dnn_model{SUFFIX}.keras")

print("\n" + "=" * 70)
print("DNN训练与评估完成!")
print("=" * 70)