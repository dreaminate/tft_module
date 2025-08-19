import pandas as pd

df1d = pd.read_csv("data/crypto_targeted_and_indicated_merged/1d/ADA_USDT_1d.csv")  # 或已在内存
ts = pd.to_datetime(df1d["timestamp"], unit="ms", utc=True)

print("unique hour-of-day on 1d:", sorted(ts.dt.hour.unique().tolist())[:10])
print("sample:", ts.head(5).tolist())
print("min/max:", ts.min(), ts.max())

# 推断步长（秒）
step = ts.sort_values().diff().dropna().dt.total_seconds().round().mode()[0]
print("inferred step seconds:", step)
