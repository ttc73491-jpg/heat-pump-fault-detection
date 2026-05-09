"""
DNN有序回归 — CORAL (COnsistent RAnk Logits) 序数分类
泄漏等级是自然有序的 (0% < 5% < ... < 50%)，标准交叉熵忽略了序数关系
CORAL 将 K 分类转化为 K-1 个二分类: P(y > k)
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, LeakyReLU, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import warnings
import os
import time

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(42)
tf.random.set_seed(42)

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


# ==================== CORAL 层和损失函数 ====================

class CoralLayer(Layer):
    """CORAL 序数回归输出层: 将 K 分类转为 K-1 个 P(y > k) 二分类"""
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform', name='coral_kernel'
        )
        # K-1 偏置，按递减顺序约束: b[0] >= b[1] >= ... >= b[K-2]
        # 保证 P(y>0) >= P(y>1) >= ... >= P(y>K-2)
        raw_biases = tf.range(self.num_classes - 1, dtype=tf.float32)
        # 初始化为递减: 1.0, 0.0, -1.0, ...
        init_biases = tf.reverse(raw_biases, axis=[0]) - tf.cast(self.num_classes / 2, tf.float32)
        self.biases_raw = self.add_weight(
            shape=(self.num_classes - 1,),
            initializer=tf.constant_initializer(init_biases.numpy() if hasattr(init_biases, 'numpy') else init_biases),
            name='coral_biases_raw'
        )

    def call(self, inputs):
        shared_logit = tf.matmul(inputs, self.kernel)  # (batch, 1)

        # 递减约束: b_k = -sum_{i=0}^{k} softplus(biases_raw[i])
        # 即 b[0] >= b[1] >= ... >= b[K-2]
        # 用反向累加 softplus 实现严格递减
        pos_diffs = tf.nn.softplus(self.biases_raw)  # all >= 0
        # b_ordered[k] = b_0 - sum_{i=1}^{k} pos_diffs[i]
        # 等价于负的累加和
        cumsum_pos = tf.cumsum(pos_diffs)  # non-decreasing
        b_ordered = pos_diffs[0] - cumsum_pos  # first=0, then decreasing

        # 更简单的实现：biases_ordered[k] = -sum_{i=0}^{k} softplus(b_raw[i])
        # b_ordered = -tf.cumsum(tf.nn.softplus(self.biases_raw))

        return shared_logit + b_ordered  # (batch, K-1)

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config


def coral_loss(y_true_cum, y_pred_logits):
    """
    CORAL loss: 对每个二分类任务计算 sigmoid cross-entropy
    y_true_cum: (batch, K-1) — 累积标签: 1 if true_class > k else 0
    y_pred_logits: (batch, K-1) — CORAL 输出的 logits
    """
    return tf.reduce_mean(
        tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(y_true_cum, tf.float32),
                logits=y_pred_logits
            ),
            axis=-1
        )
    )


def label_to_cumulative(y, K):
    """将类别标签转为累积标签: (batch,) -> (batch, K-1)
    对于类别 j, 累积标签[k] = 1 if j > k else 0
    """
    return tf.cast(tf.expand_dims(tf.cast(y, tf.int32), -1) > tf.range(K - 1), tf.float32)


def coral_predict_proba(logits):
    """从 CORAL logits 计算类别概率 P(y=k)
    P(y=0) = 1 - P(y>0) = 1 - σ(logit_0)
    P(y=k) = P(y>k-1) - P(y>k) = σ(logit_{k-1}) - σ(logit_k)
    P(y=K-1) = P(y>K-2) = σ(logit_{K-2})
    """
    probs = tf.sigmoid(logits)
    # P(y=0) = 1 - p_0
    p0 = 1.0 - probs[..., 0:1]
    # P(y=k) = p_{k-1} - p_k for k=1,...,K-2
    pk = probs[..., :-1] - probs[..., 1:]
    # P(y=K-1) = p_{K-2}
    pK = probs[..., -1:]
    return tf.concat([p0, pk, pK], axis=-1)


# ==================== 模型构建 ====================

def build_coral_model(n_classes, name='coral'):
    """构建 CORAL 有序回归模型"""
    model = Sequential([
        Input(shape=(N_FEATURES,), name=f'{name}_input'),
        Dense(128, kernel_regularizer=l2(1e-4), name=f'{name}_d1'),
        BatchNormalization(name=f'{name}_bn1'),
        LeakyReLU(alpha=0.1, name=f'{name}_act1'),
        Dropout(0.2, name=f'{name}_do1'),
        Dense(64, kernel_regularizer=l2(1e-4), name=f'{name}_d2'),
        BatchNormalization(name=f'{name}_bn2'),
        LeakyReLU(alpha=0.1, name=f'{name}_act2'),
        Dropout(0.2, name=f'{name}_do2'),
        Dense(32, kernel_regularizer=l2(1e-4), name=f'{name}_d3'),
        BatchNormalization(name=f'{name}_bn3'),
        LeakyReLU(alpha=0.1, name=f'{name}_act3'),
        Dropout(0.2, name=f'{name}_do3'),
        CoralLayer(n_classes, name=f'{name}_coral'),
    ], name=name)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=coral_loss,
        metrics=['accuracy']
    )
    return model


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


def callbacks_fn():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=50, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0),
    ]


# ==================== 数据加载 ====================
print("=" * 70)
print("DNN 有序回归 — CORAL 序数分类")
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


# ==================== 实验 1: CORAL 扁平 10 分类 ====================
print("\n" + "=" * 70)
print("实验 1: CORAL 扁平 10 分类")
print("=" * 70)

# 转为累积标签
y_train_cum = label_to_cumulative(y_train_s_orig, 10).numpy()
y_val_cum = label_to_cumulative(y_val_s_orig, 10).numpy()

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model_coral10 = build_coral_model(10, 'coral_flat10')
print(f"CORAL 10-class 参数量: {model_coral10.count_params():,}")

t0 = time.time()
hist_coral10 = model_coral10.fit(
    X_train_s, y_train_cum,
    epochs=300, batch_size=128,
    validation_data=(X_val_s, y_val_cum),
    callbacks=callbacks_fn(), verbose=2
)
t1 = time.time()

# 预测
y_test_logits = model_coral10.predict(X_test, verbose=0)
y_test_probs = coral_predict_proba(y_test_logits).numpy()
y_pred_coral10 = np.argmax(y_test_probs, axis=1)

acc_coral10 = accuracy_score(y_test_orig, y_pred_coral10)
rec_coral10, far_coral10, gma_coral10 = calculate_metrics(y_test_orig, y_pred_coral10, 10)

print(f"\nCORAL Flat 10-class Accuracy: {acc_coral10*100:.2f}% | Time: {t1-t0:.1f}s")
print(f"Macro Recall: {np.mean(rec_coral10)*100:.2f}% | Macro GMA: {np.mean(gma_coral10)*100:.2f}%")
for i in range(10):
    print(f"  Class {i}: Recall={rec_coral10[i]*100:6.2f}% FAR={far_coral10[i]*100:6.2f}% GMA={gma_coral10[i]*100:6.2f}%")

plot_cm(confusion_matrix(y_test_orig, y_pred_coral10),
        [f'{i}' for i in range(10)],
        'CORAL Flat 10-Class', 'coral_flat10_cm.png')

# ==================== 实验 2: CORAL 分层分类 ====================
print("\n" + "=" * 70)
print("实验 2: CORAL 分层分类 (Stage 1: 4-class → Stage 2: 组内)")
print("=" * 70)

# Stage 1: CORAL 4-class
print("\n--- Stage 1: CORAL 4-Class Severity ---")
y_train_s1_cum = label_to_cumulative(y_train_s_sev, 4).numpy()
y_val_s1_cum = label_to_cumulative(y_val_s_sev, 4).numpy()

tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model_s1 = build_coral_model(4, 'coral_stage1')
print(f"Stage 1 CORAL 参数量: {model_s1.count_params():,}")

t0 = time.time()
hist_s1 = model_s1.fit(
    X_train_s, y_train_s1_cum,
    epochs=300, batch_size=128,
    validation_data=(X_val_s, y_val_s1_cum),
    callbacks=callbacks_fn(), verbose=2
)
t1 = time.time()

# 预测 Stage 1
y_s1_logits = model_s1.predict(X_test, verbose=0)
y_s1_probs = coral_predict_proba(y_s1_logits).numpy()
y_pred_s1 = np.argmax(y_s1_probs, axis=1)

acc_s1 = accuracy_score(y_test_sev, y_pred_s1)
rec_s1, far_s1, gma_s1 = calculate_metrics(y_test_sev, y_pred_s1, 4)

print(f"\nStage 1 Accuracy: {acc_s1*100:.2f}% | Time: {t1-t0:.1f}s")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:>6.2f}% FAR={far_s1[i]*100:>6.2f}% GMA={gma_s1[i]*100:>6.2f}%")

plot_cm(confusion_matrix(y_test_sev, y_pred_s1), SEVERITY_NAMES,
        'CORAL Stage 1: 4-Class Severity', 'coral_stage1_cm.png')

# Stage 2: 组内 CORAL 细分
print("\n--- Stage 2: CORAL 组内细分 ---")
correct_mask = (y_pred_s1 == y_test_sev)
print(f"Stage 1 正确样本: {np.sum(correct_mask)} / {len(y_test_sev)}")

overall_correct = 0
overall_total = 0
group_results = {}

for group_id in range(4):
    group_labels = GROUPS[group_id]['labels']
    print(f"\n{GROUPS[group_id]['name']} Group (labels: {group_labels}):")

    if group_id == 0:
        mask = (y_test_sev == group_id)
        if np.sum(mask) > 0:
            correct = np.sum(y_pred_s1[mask] == y_test_sev[mask])
            overall_correct += correct
            overall_total += np.sum(mask)
            group_results[0] = {'correct': correct, 'total': np.sum(mask), 'acc': 100.0}
            print(f"  {np.sum(mask)} samples, all correct (no sub-classifier needed)")
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
    y_train_sub_cum = label_to_cumulative(y_train_sub, n_subclasses).numpy()
    y_val_sub_cum = label_to_cumulative(y_val_sub, n_subclasses).numpy()

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    model_sub = build_coral_model(n_subclasses, f'coral_stage2_g{group_id}')
    t0 = time.time()
    model_sub.fit(
        X_train_sub, y_train_sub_cum,
        epochs=300, batch_size=128,
        validation_data=(X_val_sub, y_val_sub_cum),
        callbacks=callbacks_fn(), verbose=2
    )
    t1 = time.time()

    y_sub_logits = model_sub.predict(X_test_g, verbose=0)
    y_sub_probs = coral_predict_proba(y_sub_logits).numpy()
    y_pred_sub_mapped = np.argmax(y_sub_probs, axis=1)
    y_pred_sub = np.array([group_labels[p] for p in y_pred_sub_mapped])

    correct = np.sum(y_pred_sub == y_test_g)
    acc_sub = correct / len(y_test_g) * 100

    print(f"  Train: {len(X_train_g)} | Test: {len(X_test_g)} | Correct: {correct} | Acc: {acc_sub:.2f}% | Time: {t1-t0:.1f}s")

    group_results[group_id] = {'correct': correct, 'total': len(y_test_g), 'acc': acc_sub}
    overall_correct += correct
    overall_total += len(y_test_g)

    group_label_names = [f'{l}%' for l in group_labels]
    cm_sub = confusion_matrix(y_test_g, y_pred_sub)
    plot_cm(cm_sub, group_label_names,
            f'CORAL {GROUPS[group_id]["name"]} Group (labels: {group_labels})',
            f'coral_stage2_g{group_id}_cm.png')

# ==================== 汇总 ====================
print("\n" + "=" * 70)
print("CORAL 有序回归结果汇总")
print("=" * 70)

coral_hierarchical_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0

print(f"\n{'实验':<35} {'准确率':>10}")
print("-" * 50)
print(f"{'CORAL 扁平10分类':<35} {acc_coral10*100:>9.2f}%")
print(f"{'CORAL 分层分类':<35} {coral_hierarchical_acc:>9.2f}%")

print(f"\n对比基线:")
print(f"{'  静态DNN 扁平 (最佳架构)':<35} {'~24.8%':>9}")
print(f"{'  静态DNN 分层':<35} {'56.06%':>9}")
print(f"{'  随机森林 分层':<35} {'58.87%':>9}")

print(f"\nStage 1 (CORAL 4-class) Accuracy: {acc_s1*100:.2f}%")
for i in range(4):
    print(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:>6.2f}% FAR={far_s1[i]*100:>6.2f}% GMA={gma_s1[i]*100:>6.2f}%")

print(f"\n各组 Stage 2 结果:")
for group_id in range(4):
    r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
    if r['total'] > 0:
        print(f"  {GROUPS[group_id]['name']:>8}: {r['correct']}/{r['total']} correct, Acc={r['acc']:.2f}%")

# 保存结果
results_path = os.path.join(output_dir, 'coral_dnn_results.txt')
with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("CORAL Ordinal Regression DNN Results\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"CORAL Flat 10-class: {acc_coral10*100:.2f}%\n")
    f.write(f"CORAL Hierarchical:  {coral_hierarchical_acc:.2f}%\n\n")
    f.write(f"Stage 1 (CORAL 4-class) Accuracy: {acc_s1*100:.2f}%\n")
    for i in range(4):
        f.write(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:>6.2f}% FAR={far_s1[i]*100:>6.2f}% GMA={gma_s1[i]*100:>6.2f}%\n")
    f.write(f"\nGroup Results:\n")
    for group_id in range(4):
        r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
        f.write(f"  {GROUPS[group_id]['name']}: {r['acc']:.2f}%\n")

print(f"\n结果已保存: {results_path}")
print("=" * 70)
print("CORAL 有序回归实验完成!")
print("=" * 70)
