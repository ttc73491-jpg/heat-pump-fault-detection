"""
DNN架构扫描 — 瞬态数据（仅保留前300行，21特征）
测试10种不同架构，筛选瞬态数据上DNN的最高表现
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Input, Dropout, BatchNormalization,
                                     LeakyReLU, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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

# ========== 路径配置 ==========
data_dir = r'C:\Users\ccc\Desktop\algorithm\data'
output_dir = r'C:\Users\ccc\Desktop\algorithm\11_DNN_瞬态\architecture_sweep\output'
os.makedirs(output_dir, exist_ok=True)

KEEP_FIRST_N = 300

# ========== 特征配置（瞬态Gini筛选，21特征）==========
selected_path = r'C:\Users\ccc\Desktop\algorithm\08_特征提取_瞬态\output\selected_features.csv'
SELECTED_FEATURES = pd.read_csv(selected_path)['selected_features'].tolist()

FILE_LABEL_MAP = {
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

N_CLASSES = 10
N_FEATURES = len(SELECTED_FEATURES)

# ========== 公共训练配置 ==========
COMMON_CONFIG = {
    'lr': 0.001,
    'batch_size': 64,
    'max_epochs': 300,
    'early_stop_patience': 50,
    'reduce_lr_patience': 15,
    'reduce_lr_factor': 0.5,
    'label_smoothing': 0.05,
    'l2_reg': 1e-4,
    'dropout': 0.1,
    'leaky_relu_alpha': 0.1,
    'noise_std': 0.05,
}

# ========== 标签名称 ==========
TARGET_NAMES = ['0%', '5%', '10%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']


# ==================== 数据加载 ====================

def load_data():
    print("=" * 70)
    print("数据加载 — 瞬态数据（仅保留前300行）")
    print("=" * 70)
    all_data = []
    for filename, label in FILE_LABEL_MAP.items():
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        df = df.iloc[:KEEP_FIRST_N].reset_index(drop=True)
        df['label'] = label
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    X = combined_df[SELECTED_FEATURES].values
    y = combined_df['label'].values
    X = np.nan_to_num(X, nan=0.0)
    print(f"特征: {N_FEATURES} | 样本: {len(X)} | 类别: {N_CLASSES}")
    return X, y


def preprocess(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    X_train, y_train = noise_augmented_oversample(X_train, y_train)
    y_train_oh = to_categorical(y_train, num_classes=N_CLASSES)
    y_val_oh = to_categorical(y_val, num_classes=N_CLASSES)
    y_test_oh = to_categorical(y_test, num_classes=N_CLASSES)
    print(f"训练集: {len(X_train)} (过采样后) | 验证集: {len(X_val)} | 测试集: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, y_train_oh, y_val_oh, y_test_oh


def noise_augmented_oversample(X, y):
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    X_resampled, y_resampled = [], []
    for cls, count in zip(unique_classes, counts):
        cls_indices = np.where(y == cls)[0]
        X_cls = X[cls_indices]
        if count < max_count:
            n_to_add = max_count - count
            indices_to_add = np.random.choice(cls_indices, size=n_to_add, replace=True)
            noise = np.random.normal(0, COMMON_CONFIG['noise_std'],
                                     X[indices_to_add].shape)
            X_augmented = X[indices_to_add] + noise
            X_cls = np.vstack([X_cls, X_augmented])
        X_resampled.append(X_cls)
        y_resampled.append(np.full(len(X_cls), cls))
    return np.vstack(X_resampled), np.concatenate(y_resampled)


# ==================== 架构定义 ====================

def build_dense_block(x, units, name_prefix=''):
    x = Dense(units, kernel_regularizer=l2(COMMON_CONFIG['l2_reg']),
              name=f'{name_prefix}dense_{units}')(x)
    x = BatchNormalization(name=f'{name_prefix}bn_{units}')(x)
    x = LeakyReLU(alpha=COMMON_CONFIG['leaky_relu_alpha'],
                  name=f'{name_prefix}lrelu_{units}')(x)
    x = Dropout(COMMON_CONFIG['dropout'], name=f'{name_prefix}drop_{units}')(x)
    return x


def build_residual_block(x, units, block_id):
    shortcut = x
    main = Dense(units, kernel_regularizer=l2(COMMON_CONFIG['l2_reg']),
                 name=f'res{block_id}_dense1')(x)
    main = BatchNormalization(name=f'res{block_id}_bn1')(main)
    main = LeakyReLU(alpha=COMMON_CONFIG['leaky_relu_alpha'],
                     name=f'res{block_id}_lrelu1')(main)
    main = Dropout(COMMON_CONFIG['dropout'], name=f'res{block_id}_drop1')(main)
    main = Dense(units, kernel_regularizer=l2(COMMON_CONFIG['l2_reg']),
                 name=f'res{block_id}_dense2')(main)
    main = BatchNormalization(name=f'res{block_id}_bn2')(main)
    shortcut_shape = tf.keras.backend.int_shape(shortcut)[-1]
    if shortcut_shape != units:
        shortcut = Dense(units, name=f'res{block_id}_proj')(shortcut)
    out = Add(name=f'res{block_id}_add')([main, shortcut])
    out = LeakyReLU(alpha=COMMON_CONFIG['leaky_relu_alpha'],
                    name=f'res{block_id}_lrelu2')(out)
    out = Dropout(COMMON_CONFIG['dropout'], name=f'res{block_id}_drop2')(out)
    return out


def build_architecture(arch_config):
    name = arch_config['name']
    hidden_units = arch_config['hidden_units']
    arch_type = arch_config.get('type', 'sequential')

    inputs = Input(shape=(N_FEATURES,), name=f'{name}_input')

    if arch_type == 'residual':
        x = Dense(hidden_units[0], name=f'{name}_pre_dense')(inputs)
        x = BatchNormalization(name=f'{name}_pre_bn')(x)
        x = LeakyReLU(alpha=COMMON_CONFIG['leaky_relu_alpha'],
                      name=f'{name}_pre_lrelu')(x)
        for i, units in enumerate(hidden_units):
            x = build_residual_block(x, units, block_id=i)
    else:
        x = inputs
        for i, units in enumerate(hidden_units):
            x = build_dense_block(x, units, name_prefix=f'{name}_b{i}_')

    outputs = Dense(N_CLASSES, activation='softmax', name=f'{name}_output')(x)
    model = Model(inputs=inputs, outputs=outputs, name=name)

    model.compile(
        optimizer=Adam(learning_rate=COMMON_CONFIG['lr']),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=COMMON_CONFIG['label_smoothing']),
        metrics=['accuracy']
    )
    return model


# 10种架构定义
ARCHITECTURES = [
    {'name': 'A1_Vanilla_Base',     'hidden_units': [64, 32],                    'type': 'sequential'},
    {'name': 'A2_Medium_Standard',  'hidden_units': [128, 64, 32],               'type': 'sequential'},
    {'name': 'A3_Wide_Shallow',     'hidden_units': [256, 128],                  'type': 'sequential'},
    {'name': 'A4_Deep_Funnel',      'hidden_units': [256, 128, 64, 32],          'type': 'sequential'},
    {'name': 'A5_Deep_Diamond',     'hidden_units': [64, 128, 256, 128, 64],     'type': 'sequential'},
    {'name': 'A6_Deep_Constant',    'hidden_units': [128, 128, 128, 128],        'type': 'sequential'},
    {'name': 'A7_Wide_Deep',        'hidden_units': [512, 256, 128, 64],         'type': 'sequential'},
    {'name': 'A8_Residual',         'hidden_units': [128, 128, 128],             'type': 'residual'},
    {'name': 'A9_Pyramid',          'hidden_units': [32, 64, 128, 256, 128, 64, 32], 'type': 'sequential'},
    {'name': 'A10_Narrow_Deep',     'hidden_units': [48, 48, 48, 48, 48, 48],    'type': 'sequential'},
]


# ==================== 评估函数 ====================

def evaluate_model(model, X_test, y_test, y_test_oh, arch_name):
    y_pred_onehot = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_onehot, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    recall_per_class = np.zeros(N_CLASSES)
    far_per_class = np.zeros(N_CLASSES)
    mar_per_class = np.zeros(N_CLASSES)
    specificity_per_class = np.zeros(N_CLASSES)
    gma_per_class = np.zeros(N_CLASSES)

    for i in range(N_CLASSES):
        y_true_binary = (y_test == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        TP = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        TN = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        FP = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        FN = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        recall_per_class[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        far_per_class[i] = FP / (FP + TN) if (FP + TN) > 0 else 0
        mar_per_class[i] = FN / (TP + FN) if (TP + FN) > 0 else 0
        specificity_per_class[i] = TN / (TN + FP) if (TN + FP) > 0 else 0
        gma_per_class[i] = np.sqrt(recall_per_class[i] * specificity_per_class[i])

    macro_recall = np.mean(recall_per_class)
    macro_far = np.mean(far_per_class)
    macro_mar = np.mean(mar_per_class)
    macro_gma = np.mean(gma_per_class)

    return {
        'accuracy': accuracy,
        'macro_recall': macro_recall,
        'macro_far': macro_far,
        'macro_mar': macro_mar,
        'macro_gma': macro_gma,
        'recall_per_class': recall_per_class,
        'far_per_class': far_per_class,
        'mar_per_class': mar_per_class,
        'gma_per_class': gma_per_class,
        'cm': cm,
        'y_pred': y_pred,
    }


def save_results(arch_name, results, history, train_time, stopped_epoch, y_test):
    txt_path = os.path.join(output_dir, f'sweep_transient_{arch_name}_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"DNN瞬态架构扫描 — {arch_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"特征数: {N_FEATURES} | 样本: 3000 (仅前300行)\n")
        f.write(f"训练时间: {train_time:.1f}s\n")
        f.write(f"停止轮数: {stopped_epoch}\n")
        f.write(f"总体准确率 (Accuracy): {results['accuracy'] * 100:.2f}%\n\n")
        f.write(f"宏平均召回率 (Macro Recall): {results['macro_recall'] * 100:.2f}%\n")
        f.write(f"宏平均误报率 (Macro FAR): {results['macro_far'] * 100:.2f}%\n")
        f.write(f"宏平均漏报率 (Macro MAR): {results['macro_mar'] * 100:.2f}%\n")
        f.write(f"几何平均准确率 (GMA): {results['macro_gma'] * 100:.2f}%\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'类别':<12} {'召回率':<12} {'误报率':<12} {'漏报率':<12} {'GMA':<12}\n")
        f.write("-" * 70 + "\n")
        for i in range(N_CLASSES):
            f.write(f"{i:<12} {results['recall_per_class'][i]*100:>10.2f}% "
                    f"{results['far_per_class'][i]*100:>10.2f}% "
                    f"{results['mar_per_class'][i]*100:>10.2f}% "
                    f"{results['gma_per_class'][i]*100:>10.2f}%\n")
        f.write("-" * 70 + "\n\n")
        f.write("分类报告:\n")
        f.write("-" * 70 + "\n")
        f.write(classification_report(y_test, results['y_pred'],
                                      target_names=TARGET_NAMES, digits=4))
    print(f"  结果报告: {txt_path}")

    # 混淆矩阵图
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = results['cm']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Proportion')
    ax.set_xticks(np.arange(N_CLASSES))
    ax.set_yticks(np.arange(N_CLASSES))
    ax.set_xticklabels(TARGET_NAMES, fontsize=9)
    ax.set_yticklabels(TARGET_NAMES, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm_norm.max() / 2.
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f'{cm_norm[i, j]:.1%}',
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black", fontsize=8)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'Transient {arch_name}\nAccuracy: {results["accuracy"]*100:.2f}%', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f'sweep_transient_{arch_name}_cm.png')
    plt.savefig(cm_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  混淆矩阵: {cm_path}")

    # 训练历史图
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Transient {arch_name} — Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title(f'Transient {arch_name} — Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(output_dir, f'sweep_transient_{arch_name}_history.png')
    plt.savefig(hist_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  训练曲线: {hist_path}")


# ==================== 主流程 ====================

def main():
    print("=" * 70)
    print("DNN 架构扫描 — 瞬态数据（仅前300行，21特征）")
    print("=" * 70)
    print(f"\n公共配置: lr={COMMON_CONFIG['lr']}, batch={COMMON_CONFIG['batch_size']}, "
          f"epochs<={COMMON_CONFIG['max_epochs']}")
    print(f"特征: {N_FEATURES} | 类别: {N_CLASSES} | 数据: 仅前{KEEP_FIRST_N}行")
    print(f"过采样: 噪声注入 (std={COMMON_CONFIG['noise_std']})")
    print(f"正则化: L2({COMMON_CONFIG['l2_reg']}), Dropout({COMMON_CONFIG['dropout']}), "
          f"LabelSmoothing({COMMON_CONFIG['label_smoothing']})")

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_oh, y_val_oh, y_test_oh = preprocess(X, y)

    summary = []

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=COMMON_CONFIG['early_stop_patience'],
            restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=COMMON_CONFIG['reduce_lr_factor'],
            patience=COMMON_CONFIG['reduce_lr_patience'], min_lr=1e-6, verbose=0),
    ]

    for arch in ARCHITECTURES:
        name = arch['name']
        print(f"\n{'=' * 70}")
        print(f"训练: {name} | 结构: {arch['hidden_units']} | 类型: {arch['type']}")
        print(f"{'=' * 70}")

        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        np.random.seed(42)

        model = build_architecture(arch)
        print(f"  参数量: {model.count_params():,}")

        t_start = time.time()
        history = model.fit(
            X_train, y_train_oh,
            epochs=COMMON_CONFIG['max_epochs'],
            batch_size=COMMON_CONFIG['batch_size'],
            validation_data=(X_val, y_val_oh),
            callbacks=callbacks,
            verbose=2
        )
        train_time = time.time() - t_start
        stopped_epoch = len(history.history['loss'])

        results = evaluate_model(model, X_test, y_test, y_test_oh, name)
        save_results(name, results, history, train_time, stopped_epoch, y_test)

        print(f"  准确率: {results['accuracy']*100:.2f}% | "
              f"Macro Recall: {results['macro_recall']*100:.2f}% | "
              f"GMA: {results['macro_gma']*100:.2f}%")

        summary.append({
            'architecture': name,
            'hidden_layers': str(arch['hidden_units']),
            'type': arch['type'],
            'params': model.count_params(),
            'accuracy': results['accuracy'],
            'macro_recall': results['macro_recall'],
            'macro_far': results['macro_far'],
            'macro_mar': results['macro_mar'],
            'macro_gma': results['macro_gma'],
            'train_time_s': train_time,
            'stopped_epoch': stopped_epoch,
        })

    # ========== 汇总对比 ==========
    print(f"\n{'=' * 70}")
    print("瞬态DNN架构对比汇总")
    print(f"{'=' * 70}")

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('accuracy', ascending=False)

    print(f"\n{'排名':<5} {'架构':<22} {'准确率':<10} {'Recall':<10} {'GMA':<10} {'时间(s)':<10} {'停止轮':<8}")
    print("-" * 75)
    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        print(f"{rank:<5} {row['architecture']:<22} {row['accuracy']*100:>8.2f}% "
              f"{row['macro_recall']*100:>8.2f}% {row['macro_gma']*100:>8.2f}% "
              f"{row['train_time_s']:>8.1f} {row['stopped_epoch']:>6}")

    csv_path = os.path.join(output_dir, 'transient_architecture_comparison.csv')
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总表: {csv_path}")

    plot_comparison_chart(summary_df)

    best = summary_df.iloc[0]
    print(f"\n瞬态最佳架构: {best['architecture']} ({best['accuracy']*100:.2f}%)")
    print(f"Top 3:\n{summary_df[['architecture', 'accuracy', 'macro_gma']].head(3).to_string(index=False)}")
    print(f"\n全部结果已保存至: {output_dir}")

    # 与全量数据原始架构扫描对比
    print(f"\n{'=' * 70}")
    print("瞬态 vs 全量数据 架构扫描对比")
    print(f"{'=' * 70}")
    print(f"{'架构':<22} {'瞬态准确率':<12} {'全量准确率':<12} {'变化':<10}")
    print("-" * 56)
    original_acc = {
        'A1_Vanilla_Base': 24.79, 'A2_Medium_Standard': 22.27,
        'A3_Wide_Shallow': 24.22, 'A4_Deep_Funnel': 24.55,
        'A5_Deep_Diamond': 24.20, 'A6_Deep_Constant': 24.65,
        'A7_Wide_Deep': 24.22, 'A8_Residual': 24.82,
        'A9_Pyramid': 23.53, 'A10_Narrow_Deep': 22.91,
    }
    for _, row in summary_df.sort_values('architecture').iterrows():
        name = row['architecture']
        trans_acc = row['accuracy'] * 100
        orig_acc = original_acc.get(name, 0)
        delta = trans_acc - orig_acc
        print(f"{name:<22} {trans_acc:>10.2f}% {orig_acc:>10.2f}% {delta:>+9.2f}%")


def plot_comparison_chart(summary_df):
    df = summary_df.sort_values('accuracy', ascending=True)
    names = df['architecture'].values
    acc = df['accuracy'].values * 100
    recall = df['macro_recall'].values * 100
    gma = df['macro_gma'].values * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(names))
    width = 0.25

    bars1 = ax.barh(y_pos - width, acc, width, label='Accuracy', color='#2E86AB')
    bars2 = ax.barh(y_pos, recall, width, label='Macro Recall', color='#A23B72')
    bars3 = ax.barh(y_pos + width, gma, width, label='GMA', color='#F18F01')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Score (%)', fontsize=12)
    ax.set_title('Transient DNN Architecture Sweep — Performance Comparison\n'
                 f'(21 features, First 300 rows, Noise-Aug Oversample, LabelSmoothing=0.05)', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, max(max(acc), max(recall), max(gma)) * 1.3)

    for bar, val in zip(bars1, acc):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)
    for bar, val in zip(bars3, gma):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'transient_architecture_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"对比图: {chart_path}")


if __name__ == '__main__':
    main()
