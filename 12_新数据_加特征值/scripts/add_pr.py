"""
新增压比: PR[null] = P_dis / P_suc
"""
import pandas as pd
import numpy as np
import os, warnings
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
    df = pd.read_csv(fpath)
    df['PR[null]'] = df['P_dis[bar]'] / df['P_suc[bar]']
    df.to_csv(fpath, index=False, float_format='%.15g')
    print(f"{fname}: PR min={df['PR[null]'].min():.3f}, max={df['PR[null]'].max():.3f}, mean={df['PR[null]'].mean():.3f}")

print("\n完成！")
