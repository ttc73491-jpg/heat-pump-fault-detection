"""修正 eta_is: 截断 >1.0 → 1.0, NaN → 向前填充"""
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

    col = 'eta_is[null]'
    # 截断 >1.0
    over_count = (df[col] > 1.0).sum()
    df.loc[df[col] > 1.0, col] = 1.0

    # 填充 NaN (ffill → bfill)
    nan_count = df[col].isna().sum()
    df[col] = df[col].ffill().bfill()

    df.to_csv(fpath, index=False, float_format='%.15g')
    print(f"{fname}: 截断>{over_count}, 填充NaN={nan_count}, 最终 range=[{df[col].min():.4f}, {df[col].max():.4f}]")

print("\n完成！")
