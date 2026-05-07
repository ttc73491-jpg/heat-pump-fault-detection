"""
绘制Gini特征重要性柱状图（验证版）
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

output_dir = os.path.dirname(os.path.abspath(__file__))
importance_df = pd.read_csv(os.path.join(output_dir, 'gini_feature_importance.csv'))

importance_df = importance_df[importance_df['gini_importance'] > 0].copy()

fig, ax = plt.subplots(figsize=(12, 8))

features = importance_df['feature'].values
gini_values = importance_df['gini_importance'].values * 100
cumulative = importance_df['cumulative_importance'].values * 100

colors = plt.cm.RdYlGn(gini_values / max(gini_values))
bars = ax.barh(range(len(features)), gini_values, color=colors, edgecolor='black', linewidth=0.5)

ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=10)
ax.invert_yaxis()

for i, (bar, gini, cum) in enumerate(zip(bars, gini_values, cumulative)):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{gini:.2f}%', va='center', ha='left', fontsize=9)
    ax.text(max(gini_values) * 0.6, bar.get_y() + bar.get_height()/2,
            f'({cum:.1f}%)', va='center', ha='left', fontsize=8, color='gray')

ax.set_xlabel('Gini Importance (%)', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title('Feature Importance Ranking based on Gini Impurity (Verification)\n(Heat Pump Cooling System Fault Detection)', fontsize=14)
ax.xaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.set_xlim(0, max(gini_values) * 1.25)

ax.text(0.98, 0.02, 'Cumulative importance shown in parentheses',
        transform=ax.transAxes, fontsize=9, ha='right', style='italic')

plt.tight_layout()

output_path = os.path.join(output_dir, 'feature_importance_plot.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"图片已保存至: {output_path}")
