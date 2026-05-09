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

# 8. 新特征值实验（39→15 特征，Gini 筛选后 RF/SVM/DNN）★ 最新
python 13_新特征值_特征提取/data_processing.py              # 39特征Gini筛选 → 15特征
python 14_随机森林_新特征/rf_new_features.py                # RF: 扁平98.79%, 分层99.50%
python 15_SVM_新特征/svm_new_features.py                    # SVM: 扁平98.98%, 分层99.28%
python 16_DNN_新特征/dnn_new_features.py                    # DNN: 架构扫描 + 扁平99.67%, 分层99.76%

# 9. 去V_sep_liq实验（模拟真实场景，38特征，无气液分离器液位数据）
python E_新特征值_去除气液分离器液位/18_新特征值去除液位_特征提取/data_processing.py  # 38特征Gini筛选 → 27特征
python E_新特征值_去除气液分离器液位/19_随机森林_去除液位/rf_no_vsep.py               # RF: 扁平26.86%, 分层58.85%
python E_新特征值_去除气液分离器液位/20_SVM_去除液位/svm_no_vsep.py                    # SVM: 扁平24.65%, 分层55.40%
python E_新特征值_去除气液分离器液位/21_DNN_去除液位/dnn_no_vsep.py                    # DNN: 架构扫描25.32%, 分层55.48%
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
| **DNN 新特征含V_sep_liq (15特征)** | **99.67%** | **★ 99.76%** | A8_Residual[128,128,128]，历史最佳 |
| **RF 新特征含V_sep_liq (15特征)** | 98.79% | **99.50%** | Stage 1 99.93% |
| **SVM 新特征含V_sep_liq (15特征)** | 98.98% | **99.28%** | Stage 1 99.74% |
| RF 瞬态 (21特征) | — | 69.59% | 仅用前300行，旧最佳 |
| SVM 瞬态 (21特征) | — | 60.08% | 瞬态数据 |
| DNN 瞬态分层最佳 (21特征) | 34.78% | 61.74% | [64,32]架构 |
| **RF 去V_sep_liq (27特征)** | 26.86% | **58.85%** | 模拟真实场景，≈原始RF 58.87% |
| 随机森林 (20特征) | — | 58.87% | 全量数据 |
| DNN 分层 (20特征) | ~25% | 56.06% | 分层策略使DNN从25%跃升至56.06% |
| **DNN 去V_sep_liq (27特征)** | 25.32% | **55.48%** | ≈原始DNN 56.06% |
| **SVM 去V_sep_liq (27特征)** | 24.65% | **55.40%** | ≈原始SVM 53.25% |
| SVM (20特征) | — | 53.25% | 轻度组内细分最优 (95.15%) |

**核心发现：**
- **V_sep_liq[%] 是决定性特征：** 单特征 Gini 71.26%；去除后准确率从 99.76% 断崖降至 55.48%（-44.28%）
- **实际部署瓶颈：** 无气液分离器液位数据时，38 特征上限约 **~59%**，与原始 30 特征等同
- **8 个衍生特征贡献有限：** N_comp, Valve_open, SH_suc, T_app, PR, eta_is, eta_v 无法填补 V_sep_liq 的判别力空缺
- **特征质量 > 一切：** 39→15 特征（含 V_sep_liq）直接拉到 ~99.5%；去掉后所有算法回到 ~55-59%
- **架构不重要：** 含 V_sep_liq 时最差 DNN 架构 97.74%；去 V_sep_liq 后全部坍缩在 ~25%
