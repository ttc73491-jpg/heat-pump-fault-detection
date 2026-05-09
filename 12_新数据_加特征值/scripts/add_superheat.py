"""
新增两个过热度特征：
  SH_suc[K]  = T_suc - T_sat(P_suc)     吸气过热度
  SH_evap[K] = T_eva_out - T_sat(P_eva_out)  蒸发器出口过热度
工质: CO2 (R744), 用 CoolProp 计算饱和温度
"""
import pandas as pd
import numpy as np
import CoolProp.CoolProp as CP
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

# CO2 临界压力 73.8 bar, 低于此压力才能计算饱和温度
P_CRIT_CO2 = 73.8e5  # Pa

def calc_T_sat(P_bar):
    """计算 CO2 饱和温度 [°C]，压力不在饱和区返回 NaN"""
    P_pa = P_bar * 1e5
    if P_pa <= 0 or P_pa >= P_CRIT_CO2:
        return np.nan
    try:
        T_K = CP.PropsSI('T', 'P', P_pa, 'Q', 1, 'CO2')
        return T_K - 273.15
    except:
        return np.nan

for fname in FILES:
    fpath = os.path.join(data_dir, fname)
    print(f"处理 {fname}...")
    df = pd.read_csv(fpath)

    P_suc = df['P_suc[bar]'].values
    T_suc = df['T_suc[degC]'].values
    P_eva_out = df['P_eva_out[bar]'].values
    T_eva_out = df['T_eva_out[degC]'].values

    SH_suc = np.zeros(len(df))
    SH_evap = np.zeros(len(df))
    nan_count_suc = 0
    nan_count_evap = 0

    for i in range(len(df)):
        tsat_suc = calc_T_sat(P_suc[i])
        tsat_evap = calc_T_sat(P_eva_out[i])

        if np.isnan(tsat_suc):
            SH_suc[i] = np.nan
            nan_count_suc += 1
        else:
            SH_suc[i] = T_suc[i] - tsat_suc

        if np.isnan(tsat_evap):
            SH_evap[i] = np.nan
            nan_count_evap += 1
        else:
            SH_evap[i] = T_eva_out[i] - tsat_evap

    df['SH_suc[K]'] = SH_suc
    df['SH_evap[K]'] = SH_evap

    df.to_csv(fpath, index=False, float_format='%.15g')
    print(f"  已添加 SH_suc[K], SH_evap[K]")
    if nan_count_suc > 0:
        print(f"  警告: SH_suc 有 {nan_count_suc} 个 NaN (压力超出饱和区)")
    if nan_count_evap > 0:
        print(f"  警告: SH_evap 有 {nan_count_evap} 个 NaN (压力超出饱和区)")

print("\n完成！")
