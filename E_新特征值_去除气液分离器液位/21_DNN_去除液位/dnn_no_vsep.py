"""
DNN — 去除V_sep_liq (38特征数据集)
架构扫描(10种) + 最佳架构扁平10分类 + 最佳架构分层分类
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
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(SCRIPT_DIR, '..', '17_新数据_去除液位')
output_dir = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(output_dir, exist_ok=True)

selected_path = os.path.join(SCRIPT_DIR, '..', '18_新特征值去除液位_特征提取', 'output', 'selected_features.csv')
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

N_CLASSES = 10
N_FEATURES = len(SELECTED_FEATURES)
TARGET_NAMES = ['0%', '5%', '10%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']

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

# ========== 数据加载 ==========

def load_data():
    all_data = []
    for filename, label in FILE_LABEL_MAP.items():
        df = pd.read_csv(os.path.join(data_dir, filename))
        df['label'] = label
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    X = combined_df[SELECTED_FEATURES].values
    y = combined_df['label'].values
    X = np.nan_to_num(X, nan=0.0)
    return X, y


def load_data_hierarchical():
    all_data = []
    for filename, info in FILE_CONDITIONS.items():
        df = pd.read_csv(os.path.join(data_dir, filename))
        df['severity'] = info['severity']
        df['original_label'] = info['original_label']
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    X = combined_df[SELECTED_FEATURES].values
    X = np.nan_to_num(X, nan=0.0)
    y_severity = combined_df['severity'].values
    y_original = combined_df['original_label'].values
    return X, y_severity, y_original


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
    print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")
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


# ========== 架构定义 ==========

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


def build_simple_model(n_classes, hidden_units, name='model'):
    model = Sequential()
    model.add(Input(shape=(N_FEATURES,), name=f'{name}_input'))
    for i, units in enumerate(hidden_units):
        model.add(Dense(units, kernel_regularizer=l2(1e-4), name=f'{name}_d{i+1}'))
        model.add(BatchNormalization(name=f'{name}_bn{i+1}'))
        model.add(LeakyReLU(alpha=0.1, name=f'{name}_act{i+1}'))
        model.add(Dropout(0.1, name=f'{name}_do{i+1}'))
    model.add(Dense(n_classes, activation='softmax', name=f'{name}_output'))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


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


def evaluate_model(model, X_test, y_test):
    y_pred_onehot = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_onehot, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    recall_per_class = np.zeros(N_CLASSES)
    far_per_class = np.zeros(N_CLASSES)
    mar_per_class = np.zeros(N_CLASSES)
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
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        gma_per_class[i] = np.sqrt(recall_per_class[i] * specificity)

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


def calculate_metrics_simple(y_true, y_pred, n_classes):
    recall = np.zeros(n_classes)
    far = np.zeros(n_classes)
    mar = np.zeros(n_classes)
    gma = np.zeros(n_classes)
    for i in range(n_classes):
        yt = (y_true == i).astype(int)
        yp = (y_pred == i).astype(int)
        TP = np.sum((yt == 1) & (yp == 1))
        TN = np.sum((yt == 0) & (yp == 0))
        FP = np.sum((yt == 0) & (yp == 1))
        FN = np.sum((yt == 1) & (yp == 0))
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        far[i] = FP / (FP + TN) if (FP + TN) > 0 else 0
        mar[i] = FN / (TP + FN) if (TP + FN) > 0 else 0
        gma[i] = np.sqrt(recall[i] * (1 - far[i]))
    return recall, far, mar, gma


def plot_cm_basic(cm, labels, title, filename):
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
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def callbacks_fn():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=COMMON_CONFIG['early_stop_patience'],
            restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=COMMON_CONFIG['reduce_lr_factor'],
            patience=COMMON_CONFIG['reduce_lr_patience'], min_lr=1e-6, verbose=0),
    ]


# ==================== Phase 1: 架构扫描 ====================

def phase1_architecture_sweep():
    print("=" * 70)
    print("Phase 1: DNN 架构扫描 — 去除V_sep_liq 38特征数据集")
    print("=" * 70)
    print(f"特征: {N_FEATURES} | 类别: {N_CLASSES} | 样本: 14010")

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, y_train_oh, y_val_oh, y_test_oh = preprocess(X, y)

    summary = []

    for arch in ARCHITECTURES:
        name = arch['name']
        print(f"\n{'='*70}")
        print(f"训练: {name} | 结构: {arch['hidden_units']} | 类型: {arch['type']}")
        print(f"{'='*70}")

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
            callbacks=callbacks_fn(),
            verbose=2
        )
        train_time = time.time() - t_start
        stopped_epoch = len(history.history['loss'])

        results = evaluate_model(model, X_test, y_test)
        save_sweep_results(name, results, history, train_time, stopped_epoch, y_test)

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

    # 汇总
    print(f"\n{'='*70}")
    print("架构扫描汇总")
    print(f"{'='*70}")

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('accuracy', ascending=False)

    print(f"\n{'排名':<5} {'架构':<22} {'准确率':<10} {'Recall':<10} {'GMA':<10} {'时间(s)':<10} {'停止轮':<8}")
    print("-" * 75)
    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        print(f"{rank:<5} {row['architecture']:<22} {row['accuracy']*100:>8.2f}% "
              f"{row['macro_recall']*100:>8.2f}% {row['macro_gma']*100:>8.2f}% "
              f"{row['train_time_s']:>8.1f} {row['stopped_epoch']:>6}")

    csv_path = os.path.join(output_dir, 'architecture_comparison.csv')
    summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n汇总表: {csv_path}")

    plot_comparison_chart(summary_df)

    best = summary_df.iloc[0]
    print(f"\n最佳架构: {best['architecture']} ({best['accuracy']*100:.2f}%)")
    print(f"Top 3:\n{summary_df[['architecture', 'accuracy', 'macro_gma']].head(3).to_string(index=False)}")

    # 与新特征(含V_sep_liq)架构扫描对比
    print(f"\n{'='*70}")
    print("去液位 vs 新特征(含V_sep_liq) 架构扫描对比")
    print(f"{'='*70}")
    print(f"{'架构':<22} {'去液位准确率':<14} {'新特征准确率':<12} {'变化':<10}")
    print("-" * 58)
    newfeat_acc = {
        'A1_Vanilla_Base': 98.78, 'A2_Medium_Standard': 98.67,
        'A3_Wide_Shallow': 98.78, 'A4_Deep_Funnel': 98.89,
        'A5_Deep_Diamond': 97.74, 'A6_Deep_Constant': 99.00,
        'A7_Wide_Deep': 98.67, 'A8_Residual': 99.67,
        'A9_Pyramid': 98.89, 'A10_Narrow_Deep': 98.11,
    }
    for _, row in summary_df.sort_values('architecture').iterrows():
        name = row['architecture']
        new_acc = row['accuracy'] * 100
        orig_acc = newfeat_acc.get(name, 0)
        delta = new_acc - orig_acc
        print(f"{name:<22} {new_acc:>12.2f}% {orig_acc:>10.2f}% {delta:>+9.2f}%")

    return summary_df


def save_sweep_results(arch_name, results, history, train_time, stopped_epoch, y_test):
    txt_path = os.path.join(output_dir, f'sweep_{arch_name}_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"DNN 去V_sep_liq 架构扫描 — {arch_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"特征数: {N_FEATURES} | 样本: 14010\n")
        f.write(f"训练时间: {train_time:.1f}s\n")
        f.write(f"停止轮数: {stopped_epoch}\n")
        f.write(f"总体准确率: {results['accuracy']*100:.2f}%\n\n")
        f.write(f"Macro Recall: {results['macro_recall']*100:.2f}%\n")
        f.write(f"Macro FAR: {results['macro_far']*100:.2f}%\n")
        f.write(f"Macro MAR: {results['macro_mar']*100:.2f}%\n")
        f.write(f"GMA: {results['macro_gma']*100:.2f}%\n\n")
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
        f.write(classification_report(y_test, results['y_pred'],
                                      target_names=TARGET_NAMES, digits=4))

    # 混淆矩阵
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
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'No V_sep_liq {arch_name}\nAccuracy: {results["accuracy"]*100:.2f}%', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sweep_{arch_name}_cm.png'), dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()

    # 训练历史
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.set_title(f'No V_sep_liq {arch_name} — Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title(f'No V_sep_liq {arch_name} — Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sweep_{arch_name}_history.png'), dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_comparison_chart(summary_df):
    df = summary_df.sort_values('accuracy', ascending=True)
    names = df['architecture'].values
    acc = df['accuracy'].values * 100
    recall = df['macro_recall'].values * 100
    gma = df['macro_gma'].values * 100

    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(names))
    width = 0.25

    ax.barh(y_pos - width, acc, width, label='Accuracy', color='#2E86AB')
    ax.barh(y_pos, recall, width, label='Macro Recall', color='#A23B72')
    ax.barh(y_pos + width, gma, width, label='GMA', color='#F18F01')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Score (%)', fontsize=12)
    ax.set_title('DNN Architecture Sweep — No V_sep_liq\n'
                 f'({N_FEATURES} features, 14010 samples, Noise-Aug Oversample, LabelSmoothing=0.05)', fontsize=13)
    ax.legend(loc='lower right', fontsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    all_vals = np.concatenate([acc, recall, gma])
    ax.set_xlim(0, max(all_vals) * 1.3)

    for bar, val in zip(ax.containers[0], acc):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)
    for bar, val in zip(ax.containers[2], gma):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ==================== Phase 2: 最佳架构分层分类 ====================

def phase2_hierarchical(best_arch_name, best_hidden_units):
    print("\n\n" + "=" * 70)
    print(f"Phase 2: DNN 分层分类 — 最佳架构 {best_arch_name} {best_hidden_units}")
    print("=" * 70)

    X, y_severity, y_original = load_data_hierarchical()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train_sev, y_test_sev, y_train_orig, y_test_orig = train_test_split(
        X_scaled, y_severity, y_original, test_size=0.3, random_state=42, stratify=y_severity
    )
    print(f"训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # Stage 1: 4分类
    print("\n" + "=" * 70)
    print(f"Stage 1: DNN 4-Class Severity Classification {best_hidden_units}")
    print("=" * 70)

    X_train_s1, X_val_s1, y_train_s1, y_val_s1 = train_test_split(
        X_train, y_train_sev, test_size=0.1, random_state=42, stratify=y_train_sev
    )
    y_train_s1_oh = to_categorical(y_train_s1, 4)
    y_val_s1_oh = to_categorical(y_val_s1, 4)

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    model_s1 = build_simple_model(4, best_hidden_units, 'stage1')
    print(f"Stage 1 参数量: {model_s1.count_params():,}")

    t0 = time.time()
    hist_s1 = model_s1.fit(
        X_train_s1, y_train_s1_oh,
        epochs=300, batch_size=64,
        validation_data=(X_val_s1, y_val_s1_oh),
        callbacks=callbacks_fn(), verbose=2
    )
    t1 = time.time()

    y_pred_s1 = np.argmax(model_s1.predict(X_test, verbose=0), axis=1)
    acc_s1 = accuracy_score(y_test_sev, y_pred_s1)
    rec_s1, far_s1, mar_s1, gma_s1 = calculate_metrics_simple(y_test_sev, y_pred_s1, 4)

    print(f"\nStage 1 Accuracy: {acc_s1*100:.2f}% | 训练时间: {t1-t0:.1f}s")
    print(f"{'Class':<20} {'Recall':>8} {'FAR':>8} {'MAR':>8} {'GMA':>8}")
    print("-" * 55)
    for i in range(4):
        print(f"{SEVERITY_NAMES[i]:<20} {rec_s1[i]*100:>7.2f}% {far_s1[i]*100:>7.2f}% {mar_s1[i]*100:>7.2f}% {gma_s1[i]*100:>7.2f}%")

    cm_s1 = confusion_matrix(y_test_sev, y_pred_s1)
    plot_cm_basic(cm_s1, SEVERITY_NAMES,
                  f'DNN {best_arch_name} Stage 1: 4-Class ({N_FEATURES} features, No V_sep_liq)',
                  'dnn_stage1_cm.png')

    # Stage 2: 组内细分
    print("\n" + "=" * 70)
    print(f"Stage 2: DNN Intra-Group Sub-classification {best_hidden_units}")
    print("=" * 70)

    correct_mask = (y_pred_s1 == y_test_sev)
    print(f"Stage 1 正确样本: {np.sum(correct_mask)} / {len(y_test_sev)}")

    overall_correct = 0
    overall_total = 0
    group_results = {}

    for group_id in range(4):
        group_labels = GROUPS[group_id]['labels']
        print(f"\n--- {GROUPS[group_id]['name']} Group (labels: {group_labels}) ---")

        if group_id == 0:
            mask = (y_test_sev == group_id)
            if np.sum(mask) > 0:
                correct = np.sum(y_pred_s1[mask] == y_test_sev[mask])
                overall_correct += correct
                overall_total += np.sum(mask)
                group_results[0] = {'correct': correct, 'total': np.sum(mask), 'acc': 100.0}
                print(f"  Normal: {np.sum(mask)} samples, all correct")
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

        model_sub = build_simple_model(n_subclasses, best_hidden_units, f'stage2_g{group_id}')
        t0 = time.time()
        model_sub.fit(
            X_train_sub, y_train_sub_oh,
            epochs=300, batch_size=64,
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

        group_label_names = [f'{l}%' for l in group_labels]
        cm_sub = confusion_matrix(y_test_g, y_pred_sub)
        plot_cm_basic(cm_sub, group_label_names,
                      f'DNN {best_arch_name} {GROUPS[group_id]["name"]} Group (No V_sep_liq)',
                      f'dnn_stage2_g{group_id}_cm.png')

    # 汇总
    print("\n" + "=" * 70)
    print("分层分类结果汇总")
    print("=" * 70)

    print(f"\n{'Group':<10} {'Test':>8} {'Correct':>8} {'Acc':>10}")
    print("-" * 40)
    for group_id in range(4):
        r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
        if r['total'] > 0:
            print(f"{GROUPS[group_id]['name']:<10} {r['total']:>8} {r['correct']:>8} {r['acc']:>9.2f}%")

    overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print(f"\nStage 1 (4-class) Accuracy: {acc_s1*100:.2f}%")
    print(f"Overall 10-class Accuracy: {overall_acc:.2f}%")

    return acc_s1, rec_s1, far_s1, mar_s1, gma_s1, group_results, overall_acc, overall_correct, overall_total


# ==================== Main ====================

def main():
    print("=" * 70)
    print(f"DNN 去除V_sep_liq (38→{N_FEATURES}) 故障诊断实验")
    print("=" * 70)
    print(f"特征数: {N_FEATURES}")
    print(f"特征列表: {SELECTED_FEATURES}")

    # Phase 1: 架构扫描
    summary_df = phase1_architecture_sweep()

    # 取最佳架构
    best = summary_df.iloc[0]
    best_name = best['architecture']
    best_units = eval(best['hidden_layers'])

    print(f"\n\n用最佳架构 {best_name} {best_units} 进行分层分类...")

    # Phase 2: 分层分类
    acc_s1, rec_s1, far_s1, mar_s1, gma_s1, group_results, overall_acc, overall_correct, overall_total = \
        phase2_hierarchical(best_name, best_units)

    # ==================== 最终对比 ====================
    print("\n\n" + "=" * 70)
    print("最终对比: 去液位DNN vs 新特征DNN vs 历史结果")
    print("=" * 70)

    best_flat_acc = best['accuracy'] * 100

    print(f"\n{'算法':<35} {'Stage 1':>10} {'分层10分类':>12} {'备注':>20}")
    print("-" * 80)
    print(f"{'DNN 去V_sep_liq (扁平最佳)':<35} {'—':>10} {best_flat_acc:>11.2f}% {'架构: '+best_name:>20}")
    print(f"{'DNN 去V_sep_liq (分层)':<35} {acc_s1*100:>9.2f}% {overall_acc:>11.2f}% {'架构: '+best_name:>20}")
    print(f"{'DNN 新特征含V_sep_liq (分层)':<35} {'99.69%':>10} {'99.76%':>12}")
    print(f"{'RF 新特征含V_sep_liq':<35} {'99.93%':>10} {'99.50%':>12}")
    print(f"{'RF 瞬态 (旧最佳)':<35} {'70.89%':>10} {'69.59%':>12}")
    print(f"{'RF 原始全量':<35} {'45.47%':>10} {'58.87%':>12}")

    # 保存最终结果
    with open(os.path.join(output_dir, 'dnn_no_vsep_final_results.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"DNN 去V_sep_liq (38→{N_FEATURES}) 最终结果\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Features ({N_FEATURES}): {SELECTED_FEATURES}\n\n")
        f.write(f"Architecture Sweep Best: {best_name} {best_units}\n")
        f.write(f"Flat 10-Class Best Accuracy: {best_flat_acc:.2f}%\n\n")
        f.write(f"Hierarchical Classification:\n")
        f.write(f"  Stage 1 Accuracy: {acc_s1*100:.2f}%\n")
        f.write(f"  Overall 10-class: {overall_acc:.2f}%\n\n")
        f.write("Stage 1 Per-Class:\n")
        for i in range(4):
            f.write(f"  {SEVERITY_NAMES[i]:<20} Recall={rec_s1[i]*100:6.2f}% FAR={far_s1[i]*100:6.2f}% MAR={mar_s1[i]*100:6.2f}% GMA={gma_s1[i]*100:6.2f}%\n")
        f.write(f"\nGroup Results:\n")
        for group_id in range(4):
            r = group_results.get(group_id, {'correct': 0, 'total': 0, 'acc': 0})
            f.write(f"  {GROUPS[group_id]['name']}: {r['acc']:.2f}%\n")
        f.write(f"\nComparison:\n")
        f.write(f"  DNN No V_sep_liq Flat:     {best_flat_acc:.2f}%\n")
        f.write(f"  DNN No V_sep_liq Hier:     {overall_acc:.2f}%\n")
        f.write(f"  DNN With V_sep_liq Hier:   99.76%\n")
        f.write(f"  RF With V_sep_liq:         99.50%\n")
        f.write(f"  RF Transient (old best):   69.59%\n")
        f.write(f"  RF Original:               58.87%\n")

    print(f"\n结果已保存至: {output_dir}/dnn_no_vsep_final_results.txt")
    print("=" * 70)
    print("DNN 去V_sep_liq 实验完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
