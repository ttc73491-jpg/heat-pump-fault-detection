"""
数据中心冷却系统故障检测 — 瞬态数据特征筛选
只保留前 300 行（~3000s 系统启动瞬态），与 02_特征提取 相同 Gini 方法
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os

data_dir = r'C:\Users\ccc\Desktop\algorithm\data'
output_dir = r'C:\Users\ccc\Desktop\algorithm\08_特征提取_瞬态\output'

KEEP_FIRST_N = 300

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
print(f"1. 读取数据，只保留前 {KEEP_FIRST_N} 行（瞬态数据）")
print("=" * 60)

all_data = []
for filename, label in file_label_map.items():
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    original_rows = len(df)
    df = df.iloc[:KEEP_FIRST_N].reset_index(drop=True)
    df['label'] = label
    df['condition'] = filename.replace('Heatpump_Leak_', '').replace('.csv', '')
    all_data.append(df)
    print(f"  {filename}: {original_rows} → {len(df)} 行 (仅保留前{KEEP_FIRST_N}行), 标签={label}")

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

# ==================== 三方对比：原始 vs 稳态 vs 瞬态 ====================
print("\n" + "=" * 60)
print("4. 三方对比: 原始(全量) vs 稳态(去前300) vs 瞬态(仅前300)")
print("=" * 60)

# 加载原始特征重要性
orig_imp_paths = [
    r'C:\Users\ccc\Desktop\algorithm\02_特征提取\gini_verification\gini_feature_importance.csv',
    r'C:\Users\ccc\Desktop\algorithm\02_特征提取\output\gini_feature_importance.csv',
    r'C:\Users\ccc\Desktop\algorithm\01_源数据\gini_feature_importance.csv',
]
orig_df = None
for p in orig_imp_paths:
    if os.path.exists(p):
        orig_df = pd.read_csv(p)
        break

# 加载稳态特征重要性
steady_imp_path = r'C:\Users\ccc\Desktop\algorithm\06_特征提取_稳态\output\gini_feature_importance.csv'
steady_df = pd.read_csv(steady_imp_path) if os.path.exists(steady_imp_path) else None

# 加载原始选中特征
orig_sel_paths = [
    r'C:\Users\ccc\Desktop\algorithm\02_特征提取\gini_verification\selected_features.csv',
    r'C:\Users\ccc\Desktop\algorithm\02_特征提取\output\selected_features.csv',
]
orig_selected = []
for p in orig_sel_paths:
    if os.path.exists(p):
        orig_selected = pd.read_csv(p)['selected_features'].tolist()
        break

# 加载稳态选中特征
steady_sel_path = r'C:\Users\ccc\Desktop\algorithm\06_特征提取_稳态\output\selected_features.csv'
steady_selected = pd.read_csv(steady_sel_path)['selected_features'].tolist() if os.path.exists(steady_sel_path) else []

print(f"\n{'数据集':<20} {'特征数':>8} {'Top1特征':>25} {'Top1 Gini':>10}")
print("-" * 65)
if orig_df is not None:
    orig_top1 = orig_df.iloc[0]
    print(f"{'原始(全量)':<20} {len(orig_selected):>8} {orig_top1['feature']:>25} {orig_top1['gini_importance']*100:>9.2f}%")
if steady_df is not None:
    steady_top1 = steady_df.iloc[0]
    print(f"{'稳态(去前300)':<20} {len(steady_selected):>8} {steady_top1['feature']:>25} {steady_top1['gini_importance']*100:>9.2f}%")
transient_top1 = importance_df.iloc[0]
print(f"{'瞬态(仅前300)':<20} {len(selected_features):>8} {transient_top1['feature']:>25} {transient_top1['gini_importance']*100:>9.2f}%")

print(f"\n特征交集:")
print(f"  原始 ∩ 稳态: {len(set(orig_selected) & set(steady_selected))} 共同特征")
print(f"  原始 ∩ 瞬态: {len(set(orig_selected) & set(selected_features))} 共同特征")
print(f"  稳态 ∩ 瞬态: {len(set(steady_selected) & set(selected_features))} 共同特征")
print(f"  三者共同: {len(set(orig_selected) & set(steady_selected) & set(selected_features))} 共同特征")

print(f"\n瞬态独有 (稳态无): {set(selected_features) - set(steady_selected)}")
print(f"稳态独有 (瞬态无): {set(steady_selected) - set(selected_features)}")

print("\n" + "=" * 60)
print("瞬态特征提取完成!")
print("=" * 60)
