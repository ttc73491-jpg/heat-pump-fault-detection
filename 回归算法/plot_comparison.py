import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = r'C:\Users\ccc\Desktop\algorithm\回归算法'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read metrics from all three models
algorithms = ['SVR', 'RFR', 'DNN']
train_r2 = []
test_r2 = []

for algo in algorithms:
    metrics_path = os.path.join(OUTPUT_DIR, f'0{algo}_', 'output', f'{algo.lower()}_metrics.csv')
    # Handle different numbering
    if algo == 'SVR':
        metrics_path = os.path.join(OUTPUT_DIR, '03_SVR', 'output', 'svr_metrics.csv')
    elif algo == 'RFR':
        metrics_path = os.path.join(OUTPUT_DIR, '04_RFR', 'output', 'rfr_metrics.csv')
    else:
        metrics_path = os.path.join(OUTPUT_DIR, '05_DNN', 'output', 'dnn_metrics.csv')

    df = pd.read_csv(metrics_path)
    train_r2.append(df.loc[df['Metric'] == 'R2', 'Training_Set'].values[0])
    test_r2.append(df.loc[df['Metric'] == 'R2', 'Test_Set'].values[0])

print(f'Algorithm Training R2 values:')
for algo, tr, te in zip(algorithms, train_r2, test_r2):
    print(f'  {algo}: Train={tr:.4f}, Test={te:.4f}')

# Radar chart
categories = algorithms
N = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Close the polygon

train_r2_closed = train_r2 + train_r2[:1]
test_r2_closed = test_r2 + test_r2[:1]

# Set R2 range for radar chart
r2_min = min(min(train_r2), min(test_r2)) - 0.02
r2_max = max(max(train_r2), max(test_r2)) + 0.02

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

ax.plot(angles, train_r2_closed, 'o-', color='#2196F3', linewidth=2, markersize=8,
        label='Training Set R2', markerfacecolor='#2196F3')
ax.fill(angles, train_r2_closed, alpha=0.15, color='#2196F3')

ax.plot(angles, test_r2_closed, '^--', color='#FF9800', linewidth=2, markersize=8,
        label='Test Set R2', markerfacecolor='white', markeredgecolor='#FF9800', markeredgewidth=2)
ax.fill(angles, test_r2_closed, alpha=0.15, color='#FF9800')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(algorithms, fontsize=13, fontweight='bold')
ax.set_ylim(r2_min, r2_max)
ax.set_yticks(np.linspace(r2_min, r2_max, 5))
ax.set_yticklabels([f'{x:.3f}' for x in np.linspace(r2_min, r2_max, 5)], fontsize=8)
ax.set_title('Algorithm Performance Comparison -- R2 Score\n(7 features, without V_sep_liq)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'algorithm_comparison_radar.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Radar chart saved.')
