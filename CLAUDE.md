# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于机器学习的互联网数据中心(IDC)热泵冷却系统制冷剂泄漏故障检测。数据来自仿真生成的10个泄漏等级（0%~50%），每种工况约1400个样本，30个物理特征。

## 流水线架构

```
01_源数据/ → 02_特征提取/ → 03_SVM/ / 04_随机森林/ / 05_深度神经网络/
```

- **02_特征提取/data_processing.py** 是上游依赖，必须先运行。它读取 `01_源数据/`，做 Gini 特征筛选，输出 `selected_features.csv` 和 `processed_data_with_selected_features.csv`。
- 后续三个模型目录相互独立，但都依赖 `01_源数据/` 中的原始 CSV。

## 运行脚本

所有脚本均为独立运行，无命令行参数，直接 `python <script>` 执行：

```bash
# 1. 特征提取（上游依赖）
python 02_特征提取/data_processing.py

# 2. 随机森林实验
python 04_随机森林/binary_classification.py          # 二分类
python 04_随机森林/fault_severity_classification.py   # 分层分类

# 3. SVM 实验
python 03_SVM/svm_binary.py                           # 二分类
python 03_SVM/svm_fault_severity_classification.py    # 分层分类

# 4. DNN 实验（需 tensorflow）
python 05_深度神经网络/dnn_training.py                # M9配置(16-16-8)
python 05_深度神经网络/dnn_wider_network/dnn_training.py  # 加宽网络(64-32-16)
python 05_深度神经网络/dnn_architecture_sweep.py      # 10种架构统一扫描
python 05_深度神经网络/dnn_hierarchical.py            # 分层DNN (56.06%)
python 05_深度神经网络/dnn_timeseries.py              # 时序CNN/LSTM建模
python 05_深度神经网络/dnn_coral.py                   # CORAL有序回归
python 05_深度神经网络/dnn_improved.py                # 类别加权+集成学习
```

## 项目日志

**每次操作后必须将进展追加到 `CHANGELOG.md`**，按日期分组，每条记录必须包含**确切时间（时:分）**。包括：新建/修改脚本、运行实验、发现的问题、结果数据、对比结论等。同一时间点的多个关联操作可合并为一条。不要等到用户提醒才补记。

## 关键约定

- **数据划分：** 7:3 训练/测试，`random_state=42`，`stratify=y`
- **评估指标：** Accuracy、Recall、FAR（误报率）、MAR（漏报率）、GMA（几何平均准确率）
- **分层分类分组方案：** 正常(0%) / 轻度(5%,10%,20%) / 中度(25%,30%,35%) / 重度(40%,45%,50%)
- **特征筛选：** 决策树 Gini 重要性，累计95%阈值，最低保留15个
- **DNN 过采样：** 使用自定义 `simple_random_oversample()` 而非 imbalanced-learn 库

## 当前实验结果

| 算法 | 扁平10分类 | 分层10分类 | 备注 |
|------|-----------|-----------|------|
| 随机森林 | — | **58.87%** | 当前最佳 |
| SVM | — | 53.25% | 轻度组内细分最优 (95.15%) |
| DNN 分层 | ~25% | **56.06%** | 分层策略使DNN从25%跃升至56.06% |
| DNN+RF Ensemble | — | 55.11% | 集成微调未超越基线 |
| CORAL 分层 | 18.27% | 50.63% | 序数回归无效 |
| CNN-1D 时序分层 | 13.32% | 41.54% | 时序窗口过少 |

**核心发现：**
- **架构不重要，策略才重要：** 10种DNN架构（深度2~7层、宽度48~512、含残差）全部坍缩在~25%
- **特征质量是根本瓶颈：** 所有模型Stage 1均被困在~45%，20个Gini特征无法有效分辨严重度分组
- **DNN潜力：** 当被正确路由时，轻度组内细分99.15%，证明DNN能学习细微差异
- 突破方向：特征工程（领域知识衍生特征、时序统计量），而非换模型或调参
