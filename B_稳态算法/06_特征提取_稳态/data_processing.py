"""
数据中心冷却系统故障检测 — 稳态数据特征筛选
与 02_特征提取/data_processing.py 相同方法，但剔除前 300 行（系统启动稳定过程）
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os

data_dir = r'C:\Users\ccc\Desktop\algorithm\data'
output_dir = r'C:\Users\ccc\Desktop\algorithm\06_特征提取_稳态\output'

REMOVE_FIRST_N = 300

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
print(f"1. 读取数据并剔除前 {REMOVE_FIRST_N} 行（稳态数据）")
print("=" * 60)

all_data = []
for filename, label in file_label_map.items():
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    original_rows = len(df)
    df = df.iloc[REMOVE_FIRST_N:].reset_index(drop=True)
    df['label'] = label
    df['condition'] = filename.replace('Heatpump_Leak_', '').replace('.csv', '')
    all_data.append(df)
    print(f"  {filename}: {original_rows} → {len(df)} 行 (剔除前{REMOVE_FIRST_N}行), 标签={label}")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n合并后总数据量: {len(combined_df)} 行")

feature_cols = [col for col in combined_df.columns if col not in ['#', 'time[s]', 'label', 'condition']]
X = combined_df[feature_cols]
y = combined_df['label']

print(f"特征数量: {len(feature_cols)}")
print(f"样本数量: {len(X)}")
print(f"标签分布:\n{y.value_counts().sort_index()}")

print("\n" + "=" * 60)
print("2. 决策树 Gini 重要性特征筛选")
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

print("\n特征Gini重要性排名 (全部30个):")
print(importance_df.to_string(index=False))

importance_df.to_csv(os.path.join(output_dir, 'gini_feature_importance.csv'), index=False)
print(f"\n完整特征重要性已保存至: {output_dir}/gini_feature_importance.csv")

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

selected_features_df = pd.DataFrame({'selected_features': selected_features})
selected_features_df.to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False)

X_selected = combined_df[selected_features + ['label', 'condition']]
X_selected.to_csv(os.path.join(output_dir, 'processed_data_with_selected_features.csv'), index=False)

# ==================== 与原始结果对比 ====================
print("\n" + "=" * 60)
print("4. 与原始特征筛选结果对比")
print("=" * 60)

original_features_path = r'C:\Users\ccc\Desktop\algorithm\02_特征提取\gini_verification\gini_feature_importance.csv'
if os.path.exists(original_features_path):
    orig_df = pd.read_csv(original_features_path)
elif os.path.exists(r'C:\Users\ccc\Desktop\algorithm\02_特征提取\output\gini_feature_importance.csv'):
    orig_df = pd.read_csv(r'C:\Users\ccc\Desktop\algorithm\02_特征提取\output\gini_feature_importance.csv')
else:
    orig_df = pd.read_csv(r'C:\Users\ccc\Desktop\algorithm\01_源数据\gini_feature_importance.csv')

orig_selected_path = r'C:\Users\ccc\Desktop\algorithm\02_特征提取\gini_verification\selected_features.csv'
if os.path.exists(orig_selected_path):
    orig_selected_df = pd.read_csv(orig_selected_path)
    orig_selected = orig_selected_df['selected_features'].tolist()
elif os.path.exists(r'C:\Users\ccc\Desktop\algorithm\02_特征提取\output\selected_features.csv'):
    orig_selected_df = pd.read_csv(r'C:\Users\ccc\Desktop\algorithm\02_特征提取\output\selected_features.csv')
    orig_selected = orig_selected_df['selected_features'].tolist()
else:
    orig_selected = []

# 排名对比
merged = importance_df.merge(orig_df, on='feature', suffixes=('_steady', '_orig'))
merged['rank_change'] = merged['rank_steady'] - merged['rank_orig']
merged['gini_change'] = merged['gini_importance_steady'] - merged['gini_importance_orig']

print(f"\n原始选中特征数: {len(orig_selected)}")
print(f"稳态选中特征数: {len(selected_features)}")
print(f"共同特征: {len(set(selected_features) & set(orig_selected))}")
print(f"新增 (稳态有、原始无): {set(selected_features) - set(orig_selected)}")
print(f"删除 (原始有、稳态无): {set(orig_selected) - set(selected_features)}")

print(f"\n排名变化 (按Gini差异排序):")
top_changes = merged.nlargest(10, 'gini_change')[['feature', 'rank_orig', 'rank_steady', 'rank_change', 'gini_importance_orig', 'gini_importance_steady', 'gini_change']]
print(top_changes.to_string(index=False))

print(f"\n排名上升最多的特征:")
top_risers = merged.nsmallest(10, 'rank_change')[['feature', 'rank_orig', 'rank_steady', 'rank_change', 'gini_change']]
print(top_risers.to_string(index=False))

print(f"\n排名下降最多的特征:")
top_fallers = merged.nlargest(10, 'rank_change')[['feature', 'rank_orig', 'rank_steady', 'rank_change', 'gini_change']]
print(top_fallers.to_string(index=False))

# 保存对比结果
merged.to_csv(os.path.join(output_dir, 'feature_comparison.csv'), index=False)
print(f"\n对比结果已保存至: {output_dir}/feature_comparison.csv")

print("\n" + "=" * 60)
print("稳态特征提取完成!")
print("=" * 60)
