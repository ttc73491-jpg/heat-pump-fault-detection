"""
新增气冷器出口趋近温度:
  T_app[K] = T_gc_out - T_air_in
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

data_dir = r'C:\Users\ccc\Desktop\algorithm\12_新数据_加特征值'

FILES = [
    'Heatpump_Leak_0pct.csv', 'Heatpump_Leak_5pct.csv', 'Heatpump_Leak_10pct.csv',
    'Heatpump_Leak_20pct.csv', 'Heatpump_Leak_25pct.csv', 'Heatpump_Leak_30pct.csv',
    'Heatpump_Leak_35pct.csv', 'Heatpump_Leak_40pct.csv', 'Heatpump_Leak_45pct.csv',
    'Heatpump_Leak_50pct.csv',
]

for fname in FILES:
    fpath = os.path.join(data_dir, fname)
    print(f"处理 {fname}...")
    df = pd.read_csv(fpath)

    T_gc_out = df['T_gc_out[degC]'].values
    T_air_in = df['T_air_in[degC]'].values

    df['T_app[K]'] = T_gc_out - T_air_in

    df.to_csv(fpath, index=False, float_format='%.15g')
    print(f"  已添加 T_app[K], min={df['T_app[K]'].min():.2f}, max={df['T_app[K]'].max():.2f}")

print("\n完成！")
