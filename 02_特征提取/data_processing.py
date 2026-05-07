"""
数据中心冷却系统故障检测 - 数据标注与特征筛选
参考论文：基于Gini重要性的特征选择方法
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os

# 定义数据路径
data_dir = r'C:\Users\ccc\Desktop\algorithm\01_源数据'

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

print("=" * 60)
print("1. 读取并合并数据")
print("=" * 60)

# 读取所有CSV文件
all_data = []
for filename, label in file_label_map.items():
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    df['label'] = label
    df['condition'] = filename.replace('Heatpump_Leak_', '').replace('.csv', '')
    all_data.append(df)
    print(f"  {filename}: {len(df)} 行, 标签={label}")

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n合并后总数据量: {len(combined_df)} 行")

# 分离特征和标签
feature_cols = [col for col in combined_df.columns if col not in ['#', 'time[s]', 'label', 'condition']]
X = combined_df[feature_cols]
y = combined_df['label']

print(f"特征数量: {len(feature_cols)}")
print(f"样本数量: {len(X)}")
print(f"标签分布:\n{y.value_counts().sort_index()}")

print("\n" + "=" * 60)
print("2. 使用决策树计算Gini重要性进行特征筛选")
print("=" * 60)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)

gini_importance = dt.feature_importances_

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'gini_importance': gini_importance
}).sort_values('gini_importance', ascending=False)

importance_df['cumulative_importance'] = importance_df['gini_importance'].cumsum()
importance_df['rank'] = range(1, len(importance_df) + 1)
importance_df = importance_df[['rank', 'feature', 'gini_importance', 'cumulative_importance']]

print("\n特征Gini重要性排名 (前30个):")
print(importance_df.head(30).to_string(index=False))

output_path = os.path.join(data_dir, 'gini_feature_importance.csv')
importance_df.to_csv(output_path, index=False)
print(f"\n完整特征重要性已保存至: {output_path}")

# 选择重要特征（累积重要性达到95%）
threshold = 0.95
selected_features = importance_df[importance_df['cumulative_importance'] <= threshold]['feature'].tolist()

min_features = 15
if len(selected_features) < min_features:
    selected_features = importance_df.head(min_features)['feature'].tolist()

print(f"\n" + "=" * 60)
print("3. 特征筛选结果")
print("=" * 60)
print(f"累积重要性阈值: {threshold*100}%")
print(f"筛选后特征数量: {len(selected_features)}")
print(f"\n选中的特征 ({len(selected_features)}个):")
for i, feat in enumerate(selected_features, 1):
    imp = importance_df[importance_df['feature'] == feat]['gini_importance'].values[0]
    cum_imp = importance_df[importance_df['feature'] == feat]['cumulative_importance'].values[0]
    print(f"  {i:2d}. {feat}: {imp:.6f} (累积: {cum_imp:.6f})")

# 保存选中特征
selected_features_df = pd.DataFrame({'selected_features': selected_features})
selected_features_df.to_csv(os.path.join(data_dir, 'selected_features.csv'), index=False)

# 保存处理后数据
X_selected = combined_df[selected_features + ['label', 'condition']]
X_selected.to_csv(os.path.join(data_dir, 'processed_data_with_selected_features.csv'), index=False)

print("\n" + "=" * 60)
print("处理完成!")
print("=" * 60)