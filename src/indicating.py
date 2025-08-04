# ✅ indicating_cleaned.py — 改为统一字段名（无 symbol/period 后缀），适配长格式
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from indicators import (
    calculate_ma, calculate_rsi, calculate_macd, calculate_kdj,
    calculate_momentum, calculate_vol_ma, calculate_vol_change,
    calculate_obv, calculate_bollinger, calculate_atr,
    calculate_cci, calculate_adx, calculate_vwap
)

# 1️⃣ 设定周期
timeframe = "1d"
#    python src/indicating.py 
# 2️⃣  指标开关
enabled = {
    "ma": True, "rsi": True, "macd": True, "kdj": True,
    "momentum": False, "vol_ma": False, "vol_change": True, "obv": True,
    "boll": True, "atr": True, "cci": False, "adx": False, "vwap": False,
    "lof": True
}

# 3️⃣ 路径配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_dir = os.path.join(project_root, "data", "crypto", timeframe)
output_dir = os.path.join(project_root, "data", "crypto_indicated", timeframe)
os.makedirs(output_dir, exist_ok=True)

# 4️⃣ 主循环
for fname in os.listdir(input_dir):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(input_dir, fname), parse_dates=["datetime"])
    symbol_name = fname.replace(f"_{timeframe}_all.csv", "")
    df["symbol"] = symbol_name

    print(f"📄 {fname}: {timeframe}")

    # ✅ 如果原始字段是 open_BTC_USDT_1d 格式 → 去掉后缀
    for col in ["open", "high", "low", "close", "volume"]:
        suffixed = f"{col}_{symbol_name}_{timeframe}"
        if suffixed in df.columns:
            df.rename(columns={suffixed: col}, inplace=True)

    if symbol_name == "ETH_BTC":
        out_path = os.path.join(output_dir, fname)
        df.to_csv(out_path, index=False)
        print(f"⏭️ ETH_BTC跳过指标，已重命名并保存 {out_path}")
        continue

    # 定义字段别名
    close_col = "close"
    high_col = "high"
    low_col  = "low"
    vol_col  = "volume"

    # === 技术指标计算 ===
    if enabled["ma"]:
        df["ma5"]  = calculate_ma(df, 5, column=close_col)
        df["ma10"] = calculate_ma(df, 10, column=close_col)
        df["ma20"] = calculate_ma(df, 20, column=close_col)

    if enabled["rsi"]:
        df["rsi"] = calculate_rsi(df, column=close_col)

    if enabled["macd"]:
        line, sig, hist = calculate_macd(df, column=close_col)
        df["macd_line"] = line
        df["macd_signal"] = sig
        df["macd_hist"] = hist

    if enabled["kdj"]:
        k, d, j = calculate_kdj(df, column=close_col, high_col=high_col, low_col=low_col)
        df["kdj_k"] = k
        df["kdj_d"] = d
        df["kdj_j"] = j

    if enabled["momentum"]:
        df["momentum"] = calculate_momentum(df, column=close_col)

    if enabled["vol_ma"]:
        df["vol_ma"] = calculate_vol_ma(df, column=vol_col)

    if enabled["vol_change"]:
        df["vol_change"] = calculate_vol_change(df, column=vol_col)

    if enabled["obv"]:
        df["obv"] = calculate_obv(df, column_price=close_col, column_volume=vol_col)

    if enabled["boll"]:
        ma, up, low = calculate_bollinger(df, column=close_col)
        df["boll_ma"] = ma
        df["boll_upper"] = up
        df["boll_lower"] = low

    if enabled["atr"]:
        df["atr"] = calculate_atr(df, high_col=high_col, low_col=low_col, close_col=close_col)

    if enabled["cci"]:
        df["cci"] = calculate_cci(df, high_col=high_col, low_col=low_col, close_col=close_col)

    if enabled["adx"]:
        adx, pdi, mdi = calculate_adx(df, high_col=high_col, low_col=low_col, close_col=close_col)
        df["adx"] = adx
        df["plus_di"] = pdi
        df["minus_di"] = mdi

    if enabled["vwap"]:
        df = calculate_vwap(df, column_price=close_col, column_volume=vol_col, high_col=high_col, low_col=low_col)

    # === LOF 异常检测 ===
    if enabled["lof"]:
        lof_candidates = [
            "rsi", "macd_line", "macd_hist", "kdj_j", "volume",
            "boll_ma", "boll_upper", "boll_lower", "atr", "vol_change"
        ]
        default_n_neighbors = {"1h": 24, "4h": 12, "1d": 6}.get(timeframe, 24)
        lof_contamination = 0.01

        for feat in lof_candidates:
            if feat not in df.columns:
                continue
            feat_data = df[[feat]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(feat_data) < default_n_neighbors + 1:
                print(f"⚠️ 跳过 {feat}：样本不足（{len(feat_data)} 条）")
                continue
            unique_ratio = feat_data[feat].nunique() / len(feat_data)
            dynamic_neighbors = max(default_n_neighbors, int(len(feat_data) * 0.1))

            try:
                lof = LocalOutlierFactor(n_neighbors=dynamic_neighbors, contamination=lof_contamination)
                scores = lof.fit_predict(feat_data)
                df[f"lof_score_{feat}"] = lof.negative_outlier_factor_
                df[f"is_outlier_{feat}"] = (scores == -1).astype(int)
                print(f"🧪 LOF 完成：{feat} → 异常点 {df[f'is_outlier_{feat}'].sum()} 个")
            except Exception as e:
                print(f"❌ LOF 错误：{feat} → {e}")
                df[f"lof_score_{feat}"] = np.nan
                df[f"is_outlier_{feat}"] = 0

    # === 保存结果 ===
    out_path = os.path.join(output_dir, fname)
    df.to_csv(out_path, index=False)
    print(f"✅ 保存 {out_path}")
#           python src/indicating.py