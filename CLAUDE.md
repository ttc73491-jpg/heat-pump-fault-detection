# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于机器学习的互联网数据中心(IDC)热泵冷却系统制冷剂泄漏故障检测。数据来自仿真生成的10个泄漏等级（0%~50%），每种工况约1400个样本。原始30个物理特征，经特征工程扩展至39个特征变量（含9个衍生特征）。

## 流水线架构

```
01_源数据/ → 02_特征提取/ → 03_SVM/ / 04_随机森林/ / 05_深度神经网络/
```

- **02_特征提取/data_processing.py** 是上游依赖，必须先运行。它读取 `01_源数据/`，做 Gini 特征筛选，输出 `selected_features.csv` 和 `processed_data_with_selected_features.csv`。
- 后续三个模型目录相互独立，但都依赖 `01_源数据/` 中的原始 CSV。
- **12_新数据_加特征值/** 包含扩展后的 39 特征数据（10 个工况 CSV，40 列），可直接用于后续实验。

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
python 05_深度神经网络/dnn_training_wider.py          # 加宽网络(64-32-16)
python 05_深度神经网络/dnn_architecture_sweep.py      # 10种架构统一扫描
python 05_深度神经网络/dnn_hierarchical.py            # 分层DNN (56.06%)
python 05_深度神经网络/dnn_timeseries.py              # 时序CNN/LSTM建模
python 05_深度神经网络/dnn_coral.py                   # CORAL有序回归
python 05_深度神经网络/dnn_improved.py                # 类别加权+集成学习

# 5. 稳态数据实验（剔除启动瞬态前300行）
python 06_特征提取_稳态/data_processing.py             # 稳态Gini筛选 → 18特征
python 07_随机森林_稳态/rf_steady_hierarchical.py     # 稳态RF分层 (55.90%)

# 6. 瞬态数据实验（仅保留前300行启动瞬态）★ 新最佳
python 08_特征提取_瞬态/data_processing.py             # 瞬态Gini筛选 → 21特征
python 09_随机森林_瞬态/rf_transient_hierarchical.py  # 瞬态RF分层 (69.59%)
python 10_SVM_瞬态/svm_transient_hierarchical.py      # 瞬态SVM分层 (60.08%)
python 11_DNN_瞬态/dnn_transient_hierarchical.py      # 瞬态DNN分层 (59.53%)
python 11_DNN_瞬态/architecture_sweep/dnn_transient_architecture_sweep.py  # 瞬态DNN架构扫描 (最佳34.78%)
python 11_DNN_瞬态/architecture_sweep/dnn_transient_hierarchical_best_arch.py  # 最佳架构分层 (61.74%)

# 7. 特征工程（需 CoolProp）
# 已完成，结果在 12_新数据_加特征值/，共 40 列（1 时间 + 39 特征）
python 12_新数据_加特征值/scripts/merge_features.py         # 合并3个外部参数
python 12_新数据_加特征值/scripts/add_superheat.py          # 计算过热度 (CoolProp)
python 12_新数据_加特征值/scripts/add_approach_temp.py      # 趋近温度
python 12_新数据_加特征值/scripts/add_pr.py                 # 压比
python 12_新数据_加特征值/scripts/add_isentropic_eff.py     # 等熵效率 (CoolProp)
python 12_新数据_加特征值/scripts/add_vol_eff.py            # 容积效率 (CoolProp)
```

## 项目日志

**每次操作后必须将进展追加到 `CHANGELOG.md`**，按日期分组，每条记录必须包含**确切时间（时:分）**。包括：新建/修改脚本、运行实验、发现的问题、结果数据、对比结论等。同一时间点的多个关联操作可合并为一条。不要等到用户提醒才补记。

## 关键约定

- **数据划分：** 7:3 训练/测试，`random_state=42`，`stratify=y`
- **评估指标：** Accuracy、Recall、FAR（误报率）、MAR（漏报率）、GMA（几何平均准确率）
- **分层分类分组方案：** 正常(0%) / 轻度(5%,10%,20%) / 中度(25%,30%,35%) / 重度(40%,45%,50%)
- **特征筛选：** 决策树 Gini 重要性，累计95%阈值，最低保留15个
- **CO2 物性计算：** 使用 CoolProp 7.2.0，工质 `'CO2'`。压力 bar→Pa (×1e5)，温度 °C→K (+273.15)，焓 J/kg→kJ/kg (/1000)
- **新增衍生特征（9 个）：** N_comp, Valve_open, V_sep_liq, SH_suc, SH_evap, T_app, PR, eta_is, eta_v。原始 30 特征 + 9 衍生 = 39 特征
- **DNN 过采样：** 使用自定义 `simple_random_oversample()` 而非 imbalanced-learn 库

## 当前实验结果

| 算法 | 扁平10分类 | 分层10分类 | 备注 |
|------|-----------|-----------|------|
| **RF 瞬态 (21特征)** | — | **69.59%** | ★ 新最佳，仅用前300行 |
| **SVM 瞬态 (21特征)** | — | **60.08%** | 瞬态数据，首破60% |
| **DNN 瞬态分层最佳 (21特征)** | 34.78% | **61.74%** | [64,32]架构，DNN瞬态新最佳 |
| **DNN 瞬态分层原架构 (21特征)** | — | 59.53% | [128,64,32]架构 |
| 随机森林 | — | 58.87% | 全量数据，原最佳 |
| DNN 分层 | ~25% | 56.06% | 分层策略使DNN从25%跃升至56.06% |
| RF 稳态 (18特征) | — | 55.90% | 剔除前300行，不升反降 |
| DNN+RF Ensemble | — | 55.11% | 集成微调未超越基线 |
| SVM | — | 53.25% | 轻度组内细分最优 (95.15%) |
| CORAL 分层 | 18.27% | 50.63% | 序数回归无效 |
| CNN-1D 时序分层 | 13.32% | 41.54% | 时序窗口过少 |

**核心发现：**
- **策略 > 数据 > 架构：** 全量10种+瞬态10种=20种架构全部坍缩在17-35%，换数据不如换策略（分层56%+）
- **特征质量是根本瓶颈：** 所有模型Stage 1均被困在~45%，20个Gini特征无法有效分辨严重度分组
- **DNN潜力：** 当被正确路由时，轻度组内细分99.15%，证明DNN能学习细微差异
- 突破方向：瞬态数据是关键，前300行RF达69.59%（+10.72%），SVM 60.08%，DNN 59.53%，全部超越原RF全量基线58.87%
- 稳态实验：剔除前300行瞬态后Stage 1下降5.17%，证明启动瞬态包含判别信息，不应删除
- 瞬态实验：仅用前300行（21%数据），RF Stage 1达70.89%，三算法全部超越原最佳。小样本下RF>SVM>DNN
