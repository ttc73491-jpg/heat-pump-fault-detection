"""
新增等熵效率:
  η_is[null] = (h_dis_is - h_suc) / (h_dis - h_suc)
  h_dis_is: 理想等熵排气比焓, 由 (P_dis, s_suc) 确定
  s_suc: 吸气熵, 由 (P_suc, T_suc) 确定
"""
import pandas as pd
import numpy as np
import CoolProp.CoolProp as CP
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
    print(f"处理 {fname}...")
    df = pd.read_csv(fpath)

    P_suc = df['P_suc[bar]'].values
    T_suc = df['T_suc[degC]'].values
    P_dis = df['P_dis[bar]'].values
    h_suc = df['h_suc[kJ/kg]'].values
    h_dis = df['h_dis[kJ/kg]'].values

    eta_is = np.zeros(len(df))
    nan_count = 0

    for i in range(len(df)):
        try:
            # 吸气熵 s_suc = f(P_suc, T_suc)
            s_suc = CP.PropsSI('S', 'P', P_suc[i]*1e5, 'T', T_suc[i]+273.15, 'CO2')  # J/kg·K
            # 理想等熵排气比焓 h_dis_is = f(P_dis, s_suc)
            h_dis_is = CP.PropsSI('H', 'P', P_dis[i]*1e5, 'S', s_suc, 'CO2')  # J/kg

            # 转换为 kJ/kg
            h_dis_is_kJ = h_dis_is / 1000.0
            h_suc_kJ = h_suc[i]
            h_dis_kJ = h_dis[i]

            denom = h_dis_kJ - h_suc_kJ
            if denom > 0:
                eta_is[i] = (h_dis_is_kJ - h_suc_kJ) / denom
            else:
                eta_is[i] = np.nan
                nan_count += 1
        except:
            eta_is[i] = np.nan
            nan_count += 1

    df['eta_is[null]'] = eta_is
    df.to_csv(fpath, index=False, float_format='%.15g')

    valid = eta_is[~np.isnan(eta_is)]
    print(f"  eta_is[null] 已添加, NaN={nan_count}, valid range=[{valid.min():.4f}, {valid.max():.4f}], mean={valid.mean():.4f}")

print("\n完成！")
