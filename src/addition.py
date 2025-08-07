import pandas as pd
import numpy as np
from tqdm import tqdm 
def rolling_min_max(s, window=7):
    roll_min = s.rolling(window=window, min_periods=1).min()
    roll_max = s.rolling(window=window, min_periods=1).max()
    return (s - roll_min) / (roll_max - roll_min + 1e-8)

def additional_specifics(df, window=7):
    """
    Add rolling-window Min-Max and ratio features for time-series data.
    """
    # 1. OBV: difference + rolling Min-Max
    df["obv_diff"] = df["obv"].diff().fillna(0)
    df["obv_mm"]   = rolling_min_max(df["obv_diff"], window)
    
    # 2. amplitude_range: ratio & rolling Min-Max
    df["amplitude_ratio"] = (df["high"] - df["low"]) / df["close"]
    df["amplitude_mm"]    = rolling_min_max(df["amplitude_ratio"], window)
    
    # 3. ATR: ratio & rolling Min-Max
    df["atr_ratio"] = df["atr"] / df["close"]
    df["atr_mm"]    = rolling_min_max(df["atr"], window)
    
    # 4. rolling_volatility (原 rolling_atr) ratio & rolling Min-Max
    df["rolling_vol_ratio"] = df["rolling_volatility"] / df["close"]
    df["rolling_vol_mm"]    = rolling_min_max(df["rolling_volatility"], window)
    
    # 5. Volume: log + rolling Min-Max
    df["volume_log"] = np.log1p(df["volume"])
    df["volume_mm"]  = rolling_min_max(df["volume_log"], window)
    
    # # 6. CCI: rolling Min-Max
    # df["cci_mm"] = rolling_min_max(df["cci"], window)
    
    # 7. ADX / +DI / -DI: direct ratio to [0,1]
    # df["adx_mm"]      = df["adx"] / 100
    # df["plus_di_mm"]  = df["plus_di"] / 100
    # df["minus_di_mm"] = df["minus_di"] / 100
    
    # 8. MACD Line / Signal: rolling Min-Max
    df["macd_line_mm"]   = rolling_min_max(df["macd_hist"], window)  # 用 macd_hist 作为例子
    df["macd_signal_mm"] = rolling_min_max(df["macd_hist"], window)  # 若有 macd_signal，请替换

    # 9. Trendline Slope: rolling Min-Max
    df["trendline_slope_mm"] = rolling_min_max(df["trendline_slope_20"], window)
    
    # 10. MA Diff: percentage & rolling Min-Max
    df["ma_diff_pct"] = df["ma_diff_5_20"] / (df["ma20"] + 1e-8)
    df["ma_diff_mm"]  = rolling_min_max(df["ma_diff_pct"], window)
    
    # 11. LOF Score All: rolling Min-Max
    df["lof_score_all_mm"] = rolling_min_max(df["lof_score_all"], window)
    
    # 12. ATR Slope: rolling Min-Max
    df["atr_slope_mm"] = rolling_min_max(df["atr_slope"], window)
    
    return df

def add_weekend_flag(df, date_col="timestamp", flag_col="is_weekday"):
    # 先把字符串/混合类型转成整数
    df[date_col] = pd.to_numeric(df[date_col], errors="coerce")
    # 再按毫秒级 Unix 时间戳转换
    df[date_col] = pd.to_datetime(df[date_col], unit="ms", utc=False)
    df[flag_col]  = (df[date_col].dt.weekday < 5).astype(int)
    return df

# -----------------------------
# 主流程
# -----------------------------
file_path = "data/merged/full_merged.csv"

# 1. 读取 CSV
df = pd.read_csv(file_path, parse_dates=["timestamp"])

# 2. 生成额外特征
df = additional_specifics(df, window=7)

# 3. 添加“工作日”标签（1=工作日，0=周末）
df = add_weekend_flag(df, date_col="timestamp", flag_col="is_weekday")
chunk_size = 100_000
# 4. 将结果写回原文件（不写入行索引）
total_chunks = int(np.ceil(len(df) / chunk_size))
with open(file_path, "w", newline="") as f:
    # 写入表头
    df.iloc[:0].to_csv(f, index=False)
    # 写入数据块
    for i in tqdm(range(total_chunks), desc="写入 CSV 进度", unit="chunk"):
        start = i * chunk_size
        end   = start + chunk_size
        df.iloc[start:end].to_csv(f, index=False, header=False)
print(f"[✅] 额外特征已添加并保存到 {file_path}")