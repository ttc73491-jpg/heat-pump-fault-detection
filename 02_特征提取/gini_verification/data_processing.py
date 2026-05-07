"""
数据中心冷却系统故障检测 - 数据标注与特征筛选（Gini重要性验证版）
重新运行特征筛选以验证之前结果的正确性
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import os

data_dir = r'C:\Users\ccc\Desktop\algorithm\01_源数据'
output_dir = os.path.dirname(os.path.abspath(__file__))

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

all_data = []
for filename, label in file_label_map.items():
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)
    df['label'] = label
    df['condition'] = filename.replace('Heatpump_Leak_', '').replace('.csv', '')
    all_data.append(df)
    print(f"  {filename}: {len(df)} 行, 标签={label}")

combined_df = pd.concat(all_data, ignore_index=True)
print(f"\n合并后总数据量: {len(combined_df)} 行")

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

# 被淘汰的特征
removed_features = [f for f in feature_cols if f not in selected_features]
print(f"\n被淘汰的特征 ({len(removed_features)}个):")
for feat in removed_features:
    imp = importance_df[importance_df['feature'] == feat]['gini_importance'].values[0]
    print(f"  - {feat}: {imp:.6f}")

selected_features_df = pd.DataFrame({'selected_features': selected_features})
selected_features_df.to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False)

X_selected = combined_df[selected_features + ['label', 'condition']]
X_selected.to_csv(os.path.join(output_dir, 'processed_data_with_selected_features.csv'), index=False)

# 与原始结果对比
print("\n" + "=" * 60)
print("4. 与原始结果对比验证")
print("=" * 60)

original_importance_path = r'C:\Users\ccc\Desktop\algorithm\02_特征提取\gini_feature_importance.csv'
if os.path.exists(original_importance_path):
    original_df = pd.read_csv(original_importance_path)
    original_features = original_df['feature'].tolist()
    new_features = importance_df['feature'].tolist()
    original_selected = pd.read_csv(r'C:\Users\ccc\Desktop\algorithm\02_特征提取\selected_features.csv')['selected_features'].tolist()

    # 对比特征排名
    rank_match = all(a == b for a, b in zip(original_features, new_features))
    print(f"特征排名完全一致: {'是' if rank_match else '否'}")

    if not rank_match:
        print("\n排名差异:")
        for i, (orig, new) in enumerate(zip(original_features, new_features)):
            if orig != new:
                print(f"  排名{i+1}: 原={orig} → 新={new}")

    # 对比Gini值
    max_diff = 0
    diff_count = 0
    for _, orig_row in original_df.iterrows():
        feat = orig_row['feature']
        orig_gini = orig_row['gini_importance']
        new_row = importance_df[importance_df['feature'] == feat]
        if len(new_row) > 0:
            new_gini = new_row['gini_importance'].values[0]
            diff = abs(orig_gini - new_gini)
            if diff > 1e-10:
                diff_count += 1
                if diff > max_diff:
                    max_diff = diff
                    max_diff_feat = feat

    print(f"Gini值完全一致的变量数: {30 - diff_count}/30")
    if diff_count > 0:
        print(f"最大差异: {max_diff_feat} ({max_diff:.2e})")

    # 对比选中特征
    selected_match = set(original_selected) == set(selected_features)
    print(f"\n选中特征集合一致: {'是' if selected_match else '否'}")
    if not selected_match:
        only_old = set(original_selected) - set(selected_features)
        only_new = set(selected_features) - set(original_selected)
        if only_old:
            print(f"  仅旧结果有: {only_old}")
        if only_new:
            print(f"  仅新结果有: {only_new}")

    if rank_match and diff_count == 0 and selected_match:
        print("\n>>> 结论: 新结果与原始结果完全一致，特征筛选验证通过。")
    else:
        print("\n>>> 结论: 存在差异，请检查。")
else:
    print("未找到原始结果文件，跳过对比。")

print("\n" + "=" * 60)
print("处理完成!")
print("=" * 60)
