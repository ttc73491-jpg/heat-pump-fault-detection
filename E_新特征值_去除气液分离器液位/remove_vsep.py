import os
import pandas as pd

src_dir = '../12_新数据_加特征值'
dst_dir = '.'

files = sorted([f for f in os.listdir(src_dir) if f.endswith('.csv')])
print(f"Found {len(files)} CSV files in {src_dir}")

for f in files:
    df = pd.read_csv(os.path.join(src_dir, f))
    # Remove V_sep_liq[%] column
    df = df.drop(columns=['V_sep_liq[%]'])
    out_path = os.path.join(dst_dir, f)
    df.to_csv(out_path, index=False)
    print(f"  {f}: {df.shape[1]} columns (removed V_sep_liq[%]) -> saved")

print(f"\nDone. {len(files)} files written to {os.path.abspath(dst_dir)}")
