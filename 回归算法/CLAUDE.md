# CLAUDE.md — 回归算法实验

This file provides guidance for regression-based refrigerant charge amount prediction experiments.

## 项目概述

采用回归算法（SVR、RFR、DNN）预测热泵系统制冷剂泄漏程度（连续值），替代分类算法的离散标签预测。

## 目标变量

**连续泄漏百分比**（如 0%, 5%, 10%, ..., 50%），对应充注量：

| 泄漏 | 0% | 5% | 10% | 20% | 25% | 30% | 35% | 40% | 45% | 50% |
|------|-----|-----|------|------|------|------|------|------|------|------|
| 充注量(g) | 860 | 817 | 774 | 688 | 645 | 602 | 559 | 516 | 473 | 430 |

## 目录结构

```
回归算法/
├── 01_源数据/          (10个CSV, 40列: 1时间+39特征, 来自新特征值数据)
├── 02_特征提取/        (特征筛选脚本 + output/)
├── 03_SVR/             (SVR回归 + output/)
├── 04_RFR/             (随机森林回归 + output/)
├── 05_DNN/             (DNN回归 + output/)
└── 回归算法实验报告.md
```

## 特征筛选标准流程（四阶段）

参考文档: `D:\obsidian\sardine's original vault\学习任务\毕设\回归算法特征筛选.md`

### 阶段一：物理裁剪与数据切分

**物理经验剔除（人工降维）：** 从 39 个特征中删除以下 8 列：

| 删除列 | 原因 |
|--------|------|
| `V_sep_liq[%]` (第33列) | 气液分离器液位，实际不可测 |
| `h_suc[kJ/kg]` ~ `h_eva_out[kJ/kg]` (第25-30列, 共6个) | 比焓值，冗余 |
| `T_amb[degC]` (第23列) | 环境温度 |

剩余 31 个特征进入下一步。

**数据切分：** 7:3 训练/测试划分，`random_state=42`。

### 阶段二：归一化

- 使用 `StandardScaler`
- **仅在训练集上 `fit_transform`**，测试集用 `transform`
- 防止测试集信息泄露到训练集

### 阶段三：自动化特征筛选

1. **皮尔逊相关去重：** 计算特征间相关系数矩阵，若 |r| > 0.95，删除物理意义较弱或更难测量的那个
2. **RF 重要性筛选：** 用 `RandomForestRegressor` 拟合，按累计重要性 **90%** 阈值截断（不是分类实验的 95%）

### 阶段四：最终模型训练与评估

- 仅使用筛选后的精英特征
- 重新构造 `X_train_scaled` 和 `X_test_scaled`
- 训练最终模型，计算 R²、RMSE、MAE、SD

## 算法选择要求

参考文档: `D:\obsidian\sardine's original vault\学习任务\毕设\算法的筛选.md`

### SVR 和 RFR

采用**网格搜索**（GridSearchCV）寻找最优超参数组合。
- SVR: C, gamma, epsilon
- RFR: n_estimators, max_depth, min_samples_split

### DNN

采用 **Optuna 贝叶斯优化**寻找最优架构和超参数。

**架构搜索空间:**
| 超参数 | 范围 | 采样方式 |
|--------|------|----------|
| n_layers | [1, 3] | suggest_int |
| n_units_per_layer | [16, 32, 64, 128] | suggest_categorical |
| activation | ['relu', 'elu'] | suggest_categorical |
| dropout_rate | [0.0, 0.3] | suggest_float |

**训练超参数搜索空间:**
| 超参数 | 范围 | 采样方式 |
|--------|------|----------|
| learning_rate | [1e-4, 5e-2] | suggest_float, log=True |
| batch_size | [16, 32] | suggest_categorical |

**固定配置:**
- loss: `mse`
- optimizer: `Adam`
- epochs: 150
- EarlyStopping: monitor=`val_loss`, patience=15, restore_best_weights=True

**Optuna Study:**
- direction: `minimize` (最小化验证集 MSE)
- n_trials: 50
- pruner: `MedianPruner()`

## 结果输出标准

参考文档: `D:\obsidian\sardine's original vault\学习任务\毕设\回归算法的结果输出.md`

### 执行顺序（严格执行）

1. 数据准备：读取数据，构建特征矩阵 X 和目标变量 y (泄漏百分比)
2. 数据切分：7:3 训练/测试划分
3. 标准化：`scaler.fit_transform(X_train)` → `scaler.transform(X_test)`
4. 超参数优化：网格搜索(SVR/RFR) 或 Optuna(DNN)
5. 最终拟合：用最优参数在完整训练集重训，测试集预测

### 评价指标

| 指标 | 说明 |
|------|------|
| R² | 决定系数，衡量拟合优度 |
| RMSE | 均方根误差 (√(Σ(y_i-ŷ_i)²/n)) |
| MAE | 平均绝对误差 (Σ|y_i-ŷ_i|/n) |
| SD | 预测误差标准差 |

**打印格式：**
| Metric | Training Set | Test Set |
| :--- | :--- | :--- |
| R² | x.xxxx | x.xxxx |
| RMSE | x.xxxx | x.xxxx |
| MAE | x.xxxx | x.xxxx |
| SD | x.xxxx | x.xxxx |

### 输出图像规范（DPI=300）

**图1 — 真实值 vs 预测值散点图：**
- X轴: 真实泄漏量(%)，Y轴: 预测泄漏量(%)
- y=x 理想对角线（虚线）
- 散点 alpha=0.7

**图2 — 预测误差分布直方图：**
- X轴: 预测误差 (预测值 - 真实值)
- Seaborn histplot + KDE
- x=0 处垂直基准线

**图3 — 算法性能对比雷达图：**
- 对比不同算法的训练集/测试集 R²
- 蓝色实心圆点连线(训练集) + 橙色空心三角连线(测试集)
- 半透明填充、等距同心网格线、清晰图例

### 结果分析要求

1. **泛化能力评估：** 对比训练/测试集 R² 和 RMSE，判断是否过拟合
2. **工程容差分析：** 预测误差在 ±5% 泄漏量范围内的样本占比
3. **特征贡献：** 对降低 MSE 贡献最大的前 3 个特征

## 与分类实验的关键差异

| 方面 | 分类实验 | 回归实验 |
|------|----------|----------|
| 目标变量 | 离散标签 0~9 | 连续泄漏百分比 |
| V_sep_liq | 可选（含/不含分别实验） | 强制删除（不可测） |
| 比焓列 | 保留 | 强制删除（冗余） |
| 特征筛选阈值 | 95% (Gini) | 90% (RF MSE reduction) |
| 相关去重 | 无 | Pearson |r| > 0.95 |
| DNN 调优 | 固定10架构扫描 | Optuna 贝叶斯优化 |
| DPI | 150 | 300 |
