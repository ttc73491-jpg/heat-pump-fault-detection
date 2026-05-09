"""
将3个新特征(压缩机转速、阀开度、气液分离器液位)按时间插值合并到主数据中。
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

data_dir = r'C:\Users\ccc\Desktop\algorithm\12_新数据_加特征值'

# 新特征文件 → 列名
FEATURE_FILES = {
    'com_speed.csv': 'N_comp[rpm]',
    'valve_opening_rate.csv': 'Valve_open[null]',
    'volume_per.csv': 'V_sep_liq[%]',
}

# 文件名 → 新数据中对应的列索引 (0-based, 不含时间列)
FILE_TO_COL = {
    'Heatpump_Leak_0pct.csv':  0,   # load=0.86 kg → 0%
    'Heatpump_Leak_5pct.csv':  1,   # load=0.817 kg → 5%
    'Heatpump_Leak_10pct.csv': 2,   # load=0.774 kg → 10%
    'Heatpump_Leak_20pct.csv': 3,   # load=0.688 kg → 20%
    'Heatpump_Leak_25pct.csv': 4,   # load=0.645 kg → 25%
    'Heatpump_Leak_30pct.csv': 5,   # load=0.602 kg → 30%
    'Heatpump_Leak_35pct.csv': 6,   # load=0.559 kg → 35%
    'Heatpump_Leak_40pct.csv': 7,   # load=0.516 kg → 40%
    'Heatpump_Leak_45pct.csv': 8,   # load=0.473 kg → 45%
    'Heatpump_Leak_50pct.csv': 9,   # load=0.43 kg → 50%
}


def load_new_feature(filepath):
    """读取新特征文件，返回 (time_array, data_columns_array)"""
    # 跳过前3行 (comment, header, units)
    df = pd.read_csv(filepath, skiprows=3, header=None)
    time = df.iloc[:, 0].values.astype(float)
    data = df.iloc[:, 1:].values.astype(float)  # 10 columns
    return time, data


def interpolate_value(target_time, src_time, src_values):
    """在 src_time 中查找 target_time 对应的值，线性插值"""
    idx = np.searchsorted(src_time, target_time)
    if idx == 0:
        return src_values[0]
    elif idx >= len(src_time):
        return src_values[-1]
    elif np.isclose(src_time[idx], target_time):
        return src_values[idx]
    else:
        # 线性插值: 取上下行的均值
        t_lo, t_hi = src_time[idx - 1], src_time[idx]
        v_lo, v_hi = src_values[idx - 1], src_values[idx]
        frac = (target_time - t_lo) / (t_hi - t_lo)
        return v_lo + frac * (v_hi - v_lo)


# 加载3个新特征文件
print("加载新特征文件...")
feature_data = {}
for fname, colname in FEATURE_FILES.items():
    fpath = os.path.join(data_dir, fname)
    time_arr, data_arr = load_new_feature(fpath)
    feature_data[colname] = (time_arr, data_arr)
    print(f"  {fname}: {len(time_arr)} rows, time range [{time_arr[0]:.0f}, {time_arr[-1]:.0f}]")

# 处理每个工况文件
for filename, col_idx in FILE_TO_COL.items():
    fpath = os.path.join(data_dir, filename)
    print(f"\n处理 {filename} (new data col={col_idx})...")

    df = pd.read_csv(fpath)
    times = df.iloc[:, 0].values.astype(float)

    for colname, (src_time, src_data) in feature_data.items():
        src_col = src_data[:, col_idx]
        new_values = np.array([interpolate_value(t, src_time, src_col) for t in times])
        df[colname] = new_values

    df.to_csv(fpath, index=False)
    print(f"  已添加: {list(FEATURE_FILES.values())}")
    print(f"  行数: {len(df)}, 列数: {len(df.columns)}")

print(f"\n完成！所有文件已更新。")
