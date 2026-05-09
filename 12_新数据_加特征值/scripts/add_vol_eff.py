"""
新增容积效率:
  η_v[null] = m_dot / (ρ_suc * V_th * N/60)
  ρ_suc: 吸气密度, 由 CoolProp 用 (P_suc, T_suc) 计算 [kg/m³]
  V_th = 6.8 cm³/rev = 6.8e-6 m³/rev
  N: 压缩机转速 [rpm]
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

V_TH = 6.8e-6  # m³/rev (6.8 cm³/rev)

for fname in FILES:
    fpath = os.path.join(data_dir, fname)
    print(f"处理 {fname}...")
    df = pd.read_csv(fpath)

    m_dot = df['m_dot[kg/s]'].values
    P_suc = df['P_suc[bar]'].values
    T_suc = df['T_suc[degC]'].values
    N = df['N_comp[rpm]'].values

    eta_v = np.zeros(len(df))
    nan_count = 0
    cap_count = 0

    for i in range(len(df)):
        try:
            # 吸气密度 ρ_suc = f(P_suc, T_suc) [kg/m³]
            rho_suc = CP.PropsSI('D', 'P', P_suc[i]*1e5, 'T', T_suc[i]+273.15, 'CO2')

            denom = rho_suc * V_TH * N[i] / 60.0  # kg/s
            if denom > 1e-12:
                eta_v[i] = m_dot[i] / denom
            else:
                eta_v[i] = np.nan
                nan_count += 1
        except:
            eta_v[i] = np.nan
            nan_count += 1

    # 截断 >1.0
    over_count = (eta_v > 1.0).sum()
    if over_count > 0:
        eta_v[eta_v > 1.0] = 1.0

    # 填充 NaN
    nan_after = np.isnan(eta_v).sum()
    if nan_after > 0:
        eta_v = pd.Series(eta_v).ffill().bfill().values

    df['eta_v[null]'] = eta_v
    df.to_csv(fpath, index=False, float_format='%.15g')

    valid = eta_v[~pd.isna(eta_v)] if hasattr(pd, 'isna') else eta_v
    print(f"  eta_v[null] 已添加, NaN原有={nan_count}, 截断>{over_count}, "
          f"range=[{valid.min():.4f}, {valid.max():.4f}], mean={valid.mean():.4f}")

print("\n完成！")
