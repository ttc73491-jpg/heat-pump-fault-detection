"""
数据中心冷却系统故障检测 — 39特征Gini重要性筛选
基于决策树Gini重要性进行特征选择
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

data_dir = r'C:\Users\ccc\Desktop\algorithm\12_新数据_加特征值'
output_dir = r'C:\Users\ccc\Desktop\algorithm\13_新特征值_特征提取\output'

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
print("1. 读取并合并数据 (39特征)")
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

exclude_cols = ['time[s]', 'label', 'condition']
feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
X = combined_df[feature_cols]
y = combined_df['label']

print(f"特征数量: {len(feature_cols)}")
print(f"样本数量: {len(X)}")
print(f"标签分布:\n{y.value_counts().sort_index()}")

print("\n" + "=" * 60)
print("2. 决策树Gini重要性计算")
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

print("\n特征Gini重要性排名:")
print(importance_df.to_string(index=False))

importance_df.to_csv(os.path.join(output_dir, 'gini_feature_importance.csv'), index=False)

# ---------- 绑图: Gini重要性图 ----------
plot_df = importance_df[importance_df['gini_importance'] > 0].copy()
fig, ax = plt.subplots(figsize=(14, 12))
features = plot_df['feature'].values
gini_values = plot_df['gini_importance'].values * 100
cumulative = plot_df['cumulative_importance'].values * 100

colors = plt.cm.RdYlGn(gini_values / max(gini_values))
bars = ax.barh(range(len(features)), gini_values, color=colors, edgecolor='black', linewidth=0.5)

ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=9)
ax.invert_yaxis()

for i, (bar, gini, cum) in enumerate(zip(bars, gini_values, cumulative)):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{gini:.2f}%', va='center', ha='left', fontsize=8)
    ax.text(max(gini_values) * 0.55, bar.get_y() + bar.get_height()/2,
            f'({cum:.1f}%)', va='center', ha='left', fontsize=7, color='gray')

ax.set_xlabel('Gini Importance (%)', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Feature Importance Ranking (Gini Impurity)\n39 Features — Heat Pump Fault Detection', fontsize=14)
ax.xaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.set_xlim(0, max(gini_values) * 1.3)
ax.text(0.98, 0.02, 'Cumulative importance in parentheses',
        transform=ax.transAxes, fontsize=9, ha='right', style='italic')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_plot.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n特征重要性图已保存至: {output_dir}/feature_importance_plot.png")

# ---------- 特征筛选 ----------
threshold = 0.95
selected_features = importance_df[importance_df['cumulative_importance'] <= threshold]['feature'].tolist()
min_features = 15
if len(selected_features) < min_features:
    selected_features = importance_df.head(min_features)['feature'].tolist()

print(f"\n{'='*60}")
print("3. 特征筛选结果")
print(f"{'='*60}")
print(f"累积重要性阈值: {threshold*100}%")
print(f"筛选后特征数量: {len(selected_features)}")
print(f"\n选中的特征 ({len(selected_features)}个):")
for i, feat in enumerate(selected_features, 1):
    imp = importance_df[importance_df['feature'] == feat]['gini_importance'].values[0]
    cum_imp = importance_df[importance_df['feature'] == feat]['cumulative_importance'].values[0]
    print(f"  {i:2d}. {feat}: {imp:.6f} (累积: {cum_imp:.6f})")

# 保存选中的特征列表
selected_features_df = pd.DataFrame({'selected_features': selected_features})
selected_features_df.to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False)

# 保存处理后数据
X_selected = combined_df[selected_features + ['label', 'condition']]
X_selected.to_csv(os.path.join(output_dir, 'processed_data_with_selected_features.csv'), index=False)

print(f"\n处理完成! 选中 {len(selected_features)} / {len(feature_cols)} 特征")
