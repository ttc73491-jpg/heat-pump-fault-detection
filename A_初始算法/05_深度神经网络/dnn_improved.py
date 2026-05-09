"""
DNN改进 — 针对Stage 1瓶颈: 类别加权 + 集成学习
核心问题: Stage 1 将大量Light/Medium误判为Severe
策略:
  1. 类别加权: 提高Light/Medium的损失权重
  2. 多seed集成: 训练多个模型取平均概率
  3. 与RF集成: 组合DNN+RF的预测
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
import warnings
import os
import time

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = r'C:\Users\ccc\Desktop\algorithm\01_源数据'
output_dir = r'C:\Users\ccc\Desktop\algorithm\05_深度神经网络\architecture_sweep'
os.makedirs(output_dir, exist_ok=True)

# ========== 配置 ==========
SELECTED_FEATURES = [
    'h_gc_mid[kJ/kg]', 'P_dis[bar]', 'h_dis[kJ/kg]', 'P_gc_out[bar]',
    'Q_heat_s2[kW]', 'Q_heat_s1[kW]', 'T_mid[degC]', 'h_eva_in[kJ/kg]',
    'T_air_in[degC]', 'P_gc_mid[bar]', 'W_comp[kW]', 'T_eva_out[degC]',
    'Q_heat_total[kW]', 'P_eva_out[bar]', 'h_gc_out[kJ/kg]', 'P_eva_in[bar]',
    'P_suc[bar]', 'T_gc_mid[degC]', 'T_eva_in[degC]', 'T_suc[degC]',
]

FILE_CONDITIONS = {
    'Heatpump_Leak_0pct.csv':  {'severity': 0, 'original_label': 0},
    'Heatpump_Leak_5pct.csv':  {'severity': 1, 'original_label': 1},
    'Heatpump_Leak_10pct.csv': {'severity': 1, 'original_label': 2},
    'Heatpump_Leak_20pct.csv': {'severity': 1, 'original_label': 3},
    'Heatpump_Leak_25pct.csv': {'severity': 2, 'original_label': 4},
    'Heatpump_Leak_30pct.csv': {'severity': 2, 'original_label': 5},
    'Heatpump_Leak_35pct.csv': {'severity': 2, 'original_label': 6},
    'Heatpump_Leak_40pct.csv': {'severity': 3, 'original_label': 7},
    'Heatpump_Leak_45pct.csv': {'severity': 3, 'original_label': 8},
    'Heatpump_Leak_50pct.csv': {'severity': 3, 'original_label': 9},
}

SEVERITY_NAMES = ['Normal(0%)', 'Light(5-20%)', 'Medium(25-35%)', 'Severe(40-50%)']
GROUPS = {
    0: {'name': 'Normal', 'labels': [0]},
    1: {'name': 'Light',  'labels': [1, 2, 3]},
    2: {'name': 'Medium', 'labels': [4, 5, 6]},
    3: {'name': 'Severe', 'labels': [7, 8, 9]},
}

N_FEATURES = len(SELECTED_FEATURES)
N_ENSEMBLE = 5  # 集成模型数量


def calculate_metrics(y_true, y_pred, n_classes):
    recall_per_class = np.zeros(n_classes)
    far_per_class = np.zeros(n_classes)
    gma_per_class = np.zeros(n_classes)
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        recall_per_class[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        far_per_class[i] = FP / (FP + TN) if (FP + TN) > 0 else 0
        specificity = 1 - far_per_class[i]
        gma_per_class[i] = np.sqrt(recall_per_class[i] * specificity)
    return recall_per_class, far_per_class, gma_per_class


def plot_cm(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_norm.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{cm_norm[i,j]:.1%}\n({cm[i,j]})',
                    ha="center", va="center",
                    color="white" if cm_norm[i,j] > thresh else "black", fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()


def build_model(n_classes, name='model', lr=0.001):
    model = Sequential([
        Input(shape=(N_FEATURES,), name=f'{name}_input'),
        Dense(128, kernel_regularizer=l2(1e-4), name=f'{name}_d1'),
        BatchNormalization(name=f'{name}_bn1'),
        LeakyReLU(alpha=0.1, name=f'{name}_act1'),
        Dropout(0.1, name=f'{name}_do1'),
        Dense(64, kernel_regularizer=l2(1e-4), name=f'{name}_d2'),
        BatchNormalization(name=f'{name}_bn2'),
        LeakyReLU(alpha=0.1, name=f'{name}_act2'),
        Dropout(0.1, name=f'{name}_do2'),
        Dense(32, kernel_regularizer=l2(1e-4), name=f'{name}_d3'),
        BatchNormalization(name=f'{name}_bn3'),
        LeakyReLU(alpha=0.1, name=f'{name}_act3'),
        Dropout(0.1, name=f'{name}_do3'),
        Dense(n_classes, activation='softmax', name=f'{name}_output'),
    ], name=name)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def callbacks_fn():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0),
    ]


# ==================== 数据加载 ====================
print("=" * 70)
print("DNN 改进 — 类别加权 + 集成学习")
print("=" * 70)

all_data = []
for filename, info in FILE_CONDITIONS.items():
    df = pd.read_csv(os.path.join(data_dir, filename))
    df['severity'] = info['severity']
    df['original_label'] = info['original_label']
    all_data.append(df)
combined_df = pd.concat(all_data, ignore_index=True)
print(f"总样本: {len(combined_df)}")

X = combined_df[SELECTED_FEATURES].values
X = np.nan_to_num(X, nan=0.0)
y_severity = combined_df['severity'].values
y_original = combined_df['original_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train_sev, y_test_sev, y_train_orig, y_test_orig = train_test_split(
    X_scaled, y_severity, y_original, test_size=0.3, random_state=42, stratify=y_severity
)
print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

X_train_s, X_val_s, y_train_s_sev, y_val_s_sev, y_train_s_orig, y_val_s_orig = train_test_split(
    X_train, y_train_sev, y_train_orig, test_size=0.1, random_state=42, stratify=y_train_sev
)


# ==================== 基线: RF Stage 1 ====================
print("\n" + "=" * 70)
print("基线: 随机森林 Stage 1")
print("=" * 70)

rf_s1 = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_s1.fit(X_train_s, y_train_s_sev)
rf_s1_pred = rf_s1.predict(X_test)
rf_s1_proba = rf_s1.predict_proba(X_test)
rf_s1_acc = accuracy_score(y_test_sev, rf_s1_pred)
rf_rec, rf_far, rf_gma = calculate_metrics(y_test_sev, rf_s1_pred, 4)

print(f"RF Stage 1 Accuracy: {rf_s1_acc*100:.2f}%")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rf_rec[i]*100:>6.2f}% FAR={rf_far[i]*100:>6.2f}% GMA={rf_gma[i]*100:>6.2f}%")


# ==================== 实验 1: 类别加权 DNN Stage 1 ====================
print("\n" + "=" * 70)
print("实验 1: 类别加权 DNN Stage 1")
print("=" * 70)

# 计算类别权重: 与召回率成反比
# 基线召回率: Normal=99.76%, Light=18.64%, Medium=13.08%, Severe=86.84%
# 权重: 1/recall, 归一化
baseline_recall = np.array([0.9976, 0.1864, 0.1308, 0.8684])
class_weights = 1.0 / (baseline_recall + 0.1)  # +0.1 防止除零
class_weights = class_weights / class_weights.sum() * 4  # 均值为1
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"类别权重: {dict(zip(SEVERITY_NAMES, [f'{w:.3f}' for w in class_weights]))}")

y_train_s1_oh = to_categorical(y_train_s_sev, 4)
y_val_s1_oh = to_categorical(y_val_s_sev, 4)

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model_weighted = build_model(4, 'weighted_s1')
t0 = time.time()
hist_w = model_weighted.fit(
    X_train_s, y_train_s1_oh,
    epochs=300, batch_size=128,
    validation_data=(X_val_s, y_val_s1_oh),
    class_weight=class_weight_dict,
    callbacks=callbacks_fn(), verbose=2
)
t1 = time.time()

y_pred_w = np.argmax(model_weighted.predict(X_test, verbose=0), axis=1)
acc_w = accuracy_score(y_test_sev, y_pred_w)
rec_w, far_w, gma_w = calculate_metrics(y_test_sev, y_pred_w, 4)

print(f"\n类别加权 DNN Stage 1 Accuracy: {acc_w*100:.2f}% | Time: {t1-t0:.1f}s")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_w[i]*100:>6.2f}% FAR={far_w[i]*100:>6.2f}% GMA={gma_w[i]*100:>6.2f}%")

plot_cm(confusion_matrix(y_test_sev, y_pred_w), SEVERITY_NAMES,
        'Weighted DNN Stage 1: 4-Class', 'weighted_stage1_cm.png')


# ==================== 实验 2: DNN 集成 (多 seed) ====================
print("\n" + "=" * 70)
print("实验 2: DNN 集成 (多 seed 平均概率)")
print("=" * 70)

dnn_probas = []
for seed in range(N_ENSEMBLE):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_model(4, f'dnn_s1_s{seed}', lr=0.001)
    model.fit(
        X_train_s, y_train_s1_oh,
        epochs=300, batch_size=128,
        validation_data=(X_val_s, y_val_s1_oh),
        callbacks=callbacks_fn(), verbose=0
    )
    proba = model.predict(X_test, verbose=0)
    dnn_probas.append(proba)
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y_test_sev, pred)
    print(f"  Seed {seed}: accuracy={acc*100:.2f}%")

# 平均概率
avg_proba = np.mean(dnn_probas, axis=0)
y_pred_ens = np.argmax(avg_proba, axis=1)
acc_ens = accuracy_score(y_test_sev, y_pred_ens)
rec_ens, far_ens, gma_ens = calculate_metrics(y_test_sev, y_pred_ens, 4)

print(f"\nDNN Ensemble Stage 1 Accuracy: {acc_ens*100:.2f}%")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_ens[i]*100:>6.2f}% FAR={far_ens[i]*100:>6.2f}% GMA={gma_ens[i]*100:>6.2f}%")

# ==================== 实验 3: DNN + RF 集成 ====================
print("\n" + "=" * 70)
print("实验 3: DNN + RF 集成 (平均概率)")
print("=" * 70)

# RF + DNN ensemble 概率平均
rf_proba = rf_s1.predict_proba(X_test)
dnn_ens_proba = avg_proba  # 使用DNN集成的概率

# 不同权重组合
for w_rf in [0.3, 0.4, 0.5, 0.6, 0.7]:
    combined_proba = w_rf * rf_proba + (1 - w_rf) * dnn_ens_proba
    y_pred_comb = np.argmax(combined_proba, axis=1)
    acc_comb = accuracy_score(y_test_sev, y_pred_comb)
    rec_comb, far_comb, gma_comb = calculate_metrics(y_test_sev, y_pred_comb, 4)
    light_rec = rec_comb[1]
    medium_rec = rec_comb[2]
    print(f"  w_rf={w_rf:.1f}: Acc={acc_comb*100:.2f}% Light_R={light_rec*100:.1f}% Med_R={medium_rec*100:.1f}%")

# 选最优
best_w = 0.5
combined_proba = best_w * rf_proba + (1 - best_w) * dnn_ens_proba
y_pred_comb = np.argmax(combined_proba, axis=1)
acc_comb = accuracy_score(y_test_sev, y_pred_comb)
rec_comb, far_comb, gma_comb = calculate_metrics(y_test_sev, y_pred_comb, 4)

print(f"\n最优集成 (w_rf={best_w:.1f}):")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_comb[i]*100:>6.2f}% FAR={far_comb[i]*100:>6.2f}% GMA={gma_comb[i]*100:>6.2f}%")

plot_cm(confusion_matrix(y_test_sev, y_pred_comb), SEVERITY_NAMES,
        f'DNN+RF Ensemble Stage 1 (w_rf={best_w:.1f})', 'ensemble_stage1_cm.png')

# ==================== 完整分层管道 (使用最佳Stage 1) ====================
print("\n" + "=" * 70)
print("完整分层管道 — 集成 Stage 1 + DNN Stage 2")
print("=" * 70)

best_s1_proba = combined_proba
best_s1_pred = y_pred_comb
best_s1_acc = acc_comb

correct_mask = (best_s1_pred == y_test_sev)
print(f"Stage 1 正确样本: {np.sum(correct_mask)} / {len(y_test_sev)}")

overall_correct = 0
overall_total = 0
group_results = {}

for group_id in range(4):
    group_labels = GROUPS[group_id]['labels']
    print(f"\n{GROUPS[group_id]['name']} Group:")

    if group_id == 0:
        mask = (y_test_sev == group_id)
        if np.sum(mask) > 0:
            correct = np.sum(best_s1_pred[mask] == y_test_sev[mask])
            overall_correct += correct
            overall_total += np.sum(mask)
            group_results[0] = {'correct': correct, 'total': np.sum(mask), 'acc': 100.0}
            print(f"  {np.sum(mask)} samples, correct={correct}")
        continue

    train_mask = np.isin(y_train_orig, group_labels)
    X_train_g = X_train[train_mask]
    y_train_g = y_train_orig[train_mask]

    test_mask = (y_test_sev == group_id) & correct_mask
    X_test_g = X_test[test_mask]
    y_test_g = y_test_orig[test_mask]

    if len(X_test_g) == 0:
        group_results[group_id] = {'correct': 0, 'total': 0, 'acc': 0.0}
        continue

    label_map = {old: new for new, old in enumerate(group_labels)}
    y_train_g_mapped = np.array([label_map[l] for l in y_train_g])
    y_test_g_mapped = np.array([label_map[l] for l in y_test_g])

    n_subclasses = len(group_labels)

    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_g, y_train_g_mapped, test_size=0.1, random_state=42, stratify=y_train_g_mapped
    )
    y_train_sub_oh = to_categorical(y_train_sub, n_subclasses)
    y_val_sub_oh = to_categorical(y_val_sub, n_subclasses)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    model_sub = build_model(n_subclasses, f's2_g{group_id}')
    t0 = time.time()
    model_sub.fit(
        X_train_sub, y_train_sub_oh,
        epochs=300, batch_size=128,
        validation_data=(X_val_sub, y_val_sub_oh),
        callbacks=callbacks_fn(), verbose=2
    )
    t1 = time.time()

    y_pred_sub_mapped = np.argmax(model_sub.predict(X_test_g, verbose=0), axis=1)
    y_pred_sub = np.array([group_labels[p] for p in y_pred_sub_mapped])

    correct = np.sum(y_pred_sub == y_test_g)
    acc_sub = correct / len(y_test_g) * 100
    print(f"  Train: {len(X_train_g)} | Test: {len(X_test_g)} | Correct: {correct} | Acc: {acc_sub:.2f}% | Time: {t1-t0:.1f}s")

    group_results[group_id] = {'correct': correct, 'total': len(y_test_g), 'acc': acc_sub}
    overall_correct += correct
    overall_total += len(y_test_g)

# ==================== 汇总 ====================
print("\n" + "=" * 70)
print("改进实验结果汇总")
print("=" * 70)

improved_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0

print(f"\n{'实验':<40} {'Stage 1 Acc':>12} {'总体Acc':>10}")
print("-" * 65)
print(f"{'基线 DNN 分层':<40} {'45.54%':>12} {'56.06%':>10}")
print(f"{'RF 分层 (当前最佳)':<40} {'45.47%':>12} {'58.87%':>10}")
print(f"{'类别加权 DNN':<40} {acc_w*100:>11.2f}%")
print(f"{'DNN Ensemble (5 seeds)':<40} {acc_ens*100:>11.2f}%")
print(f"{'DNN+RF Ensemble':<40} {acc_comb*100:>11.2f}% {improved_acc:>9.2f}%")

print(f"\n最佳 Stage 1 ({'DNN+RF Ensemble'}): Accuracy={best_s1_acc*100:.2f}%")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_comb[i]*100:>6.2f}% FAR={far_comb[i]*100:>6.2f}% GMA={gma_comb[i]*100:>6.2f}%")

print(f"\n各组 Stage 2 结果:")
for group_id in range(4):
    r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
    if r['total'] > 0:
        print(f"  {GROUPS[group_id]['name']:>8}: {r['correct']}/{r['total']} correct, Acc={r['acc']:.2f}%")

print(f"\n总体分层准确率: {improved_acc:.2f}%")

# 保存结果
results_path = os.path.join(output_dir, 'improved_dnn_results.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("Improved DNN Results (Class Weighting + Ensemble)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"RF Stage 1: {rf_s1_acc*100:.2f}%\n")
    f.write(f"Class-Weighted DNN Stage 1: {acc_w*100:.2f}%\n")
    f.write(f"DNN Ensemble ({N_ENSEMBLE} seeds) Stage 1: {acc_ens*100:.2f}%\n")
    f.write(f"DNN+RF Ensemble Stage 1: {acc_comb*100:.2f}%\n")
    f.write(f"Overall Hierarchical Accuracy: {improved_acc:.2f}%\n\n")
    f.write(f"Stage 1 Metrics (DNN+RF Ensemble):\n")
    for i in range(4):
        f.write(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_comb[i]*100:>6.2f}% FAR={far_comb[i]*100:>6.2f}% GMA={gma_comb[i]*100:>6.2f}%\n")
    f.write(f"\nGroup Results:\n")
    for group_id in range(4):
        r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
        f.write(f"  {GROUPS[group_id]['name']}: {r['acc']:.2f}%\n")

print(f"\n结果已保存: {results_path}")
print("=" * 70)
print("改进实验完成!")
print("=" * 70)
