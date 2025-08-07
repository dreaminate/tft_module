import os
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# === 自研指标库（新） ===
from indicators import (
    # 原有基础指标
    calculate_ma, calculate_rsi, calculate_macd, calculate_kdj,
    calculate_momentum, calculate_vol_ma, calculate_vol_change,
    calculate_obv, calculate_bollinger, calculate_atr,
    calculate_cci, calculate_adx, calculate_vwap,
    # 新增信号/扩展指标
    macd_bullish_cross, rsi_overbought, rsi_oversold, price_above_ma,
    breakout_recent_high, pullback_from_high, trendline_slope, ma_diff,
    volume_relative, volume_percentile, support_level, resistance_level,
    distance_to_support, distance_to_resistance,
    volume_spike_near_resist, volume_spike_near_support,
)

# 1️⃣ 设定周期（脚本运行时可通过 CLI --timeframe 覆盖）
timeframe = "1h"
 # python src/indicating.py --timeframe 1h
# 2️⃣ 指标开关（新增字段已补充，可按需开启/关闭）
enabled = {
    # === 通用技术指标 ===
    "ma": True,
    "rsi": True,
    "macd": True,
    "kdj": True,
    "momentum": True,
    "vol_ma": True,
    "vol_change": True,
    "obv": True,
    "boll": True,
    "atr": True,
    "cci": False,
    "adx": False,
    "vwap": False,
    # === 新交易信号 ===
    "signal_macd_cross": True,
    "signal_rsi_ob_os": True,
    "signal_price_above_ma": True,
    "signal_breakout_pullback": True,
    # === 趋势/量能/支撑阻力 ===
    "trend_strength": True,
    "volume_features": True,
    "support_resistance": True,
    # === LOF 异常 ===
    "lof": True,
}

# 3️⃣ 路径配置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_dir = os.path.join(project_root, "data", "crypto", timeframe)
output_dir = os.path.join(project_root, "data", "crypto_indicated", timeframe)
os.makedirs(output_dir, exist_ok=True)

# 4️⃣ 主循环遍历 symbol CSV
for fname in os.listdir(input_dir):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(input_dir, fname), parse_dates=["datetime"])
    symbol_name = fname.replace(f"_{timeframe}_all.csv", "")
    df["symbol"] = symbol_name

    print(f"📄 {fname}: {timeframe}")

    # ✅ 如果原始字段带后缀 → 去掉，统一字段名 (open/high/low/close/volume)
    for col in ["open", "high", "low", "close", "volume"]:
        suffixed = f"{col}_{symbol_name}_{timeframe}"
        if suffixed in df.columns:
            df.rename(columns={suffixed: col}, inplace=True)

    # === 字段别名 ===
    close_col = "close"
    high_col = "high"
    low_col = "low"
    vol_col = "volume"

    # =========================
    # 技术指标 (原有)
    # =========================
    if enabled["ma"]:
        df["ma5"] = calculate_ma(df, 5, column=close_col)
        df["ma20"] = calculate_ma(df, 20, column=close_col)
    if enabled["rsi"]:
        df["rsi"] = calculate_rsi(df, column=close_col)
    if enabled["macd"]:
        _, _, hist = calculate_macd(df, column=close_col)
        df["macd_hist"] = hist
    if enabled["kdj"]:
        k, _, j = calculate_kdj(df, column=close_col, high_col=high_col, low_col=low_col)
        df["kdj_k"] = k
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
        _, up, lowbb = calculate_bollinger(df, column=close_col)
        df["boll_upper"] = up
        df["boll_lower"] = lowbb
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
        df["vwap"] = calculate_vwap(df, column_price=close_col, column_volume=vol_col, high_col=high_col, low_col=low_col)

    # =========================
    # 新增交易信号
    # =========================
    if enabled["signal_macd_cross"] and "macd_hist" in df.columns:
        df["macd_bullish_cross"] = macd_bullish_cross(df)
    if enabled["signal_rsi_ob_os"] and "rsi" in df.columns:
        df["rsi_overbought"] = rsi_overbought(df)
        df["rsi_oversold"] = rsi_oversold(df)
    if enabled["signal_price_above_ma"] and {"close", "ma20"}.issubset(df.columns):
        df["price_above_ma20"] = price_above_ma(df, ma_col="ma20")
    if enabled["signal_breakout_pullback"] and {"high", "close"}.issubset(df.columns):
        df["breakout_recent_high"] = breakout_recent_high(df)
        df["pullback_from_high"] = pullback_from_high(df)

    # =========================
    # 趋势/量能/支撑阻力
    # =========================
    if enabled["trend_strength"] and {"close"}.issubset(df.columns):
        df["trendline_slope_20"] = trendline_slope(df)
        if {"ma5", "ma20"}.issubset(df.columns):
            df["ma_diff_5_20"] = ma_diff(df)
    if enabled["volume_features"]:
        df["volume_relative_20"] = volume_relative(df)
        df["volume_percentile"] = volume_percentile(df)
    if enabled["support_resistance"] :
        df["support_level_20"] = support_level(df, low_col=low_col)
        df["resistance_level_20"] = resistance_level(df, high_col=high_col)
        df["distance_to_support"] = distance_to_support(df)
        df["distance_to_resistance"] = distance_to_resistance(df)
        if {"volume_relative_20", "distance_to_support", "distance_to_resistance"}.issubset(df.columns):
            df["volume_spike_near_support"] = volume_spike_near_support(df)
            df["volume_spike_near_resist"] = volume_spike_near_resist(df)

    # =========================
    # LOF 本地异常因子
    # =========================
    if enabled["lof"]:
        lof_candidates = [
            "rsi", "macd_hist", "kdj_j", "volume", "boll_upper", "boll_lower", "atr", "vol_change",
            "pullback_from_high", "volume_relative_20", "distance_to_support", "distance_to_resistance",
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
            dynamic_neighbors = max(default_n_neighbors, int(len(feat_data) * 0.1))
            try:
                lof = LocalOutlierFactor(n_neighbors=dynamic_neighbors, contamination=lof_contamination)
                scores = lof.fit_predict(feat_data)
                df[f"lof_score_{feat}"] = lof.negative_outlier_factor_
                df[f"is_outlier_{feat}"] = (scores == -1).astype(int)
                print(f"🧪 LOF 完成：{feat} → 异常 {df[f'is_outlier_{feat}'].sum()} 条")
            except Exception as e:
                print(f"❌ LOF 错误：{feat} → {e}")
                df[f"lof_score_{feat}"] = np.nan
                df[f"is_outlier_{feat}"] = 0

    # =========================
    # 保存输出
    # =========================
    out_path = os.path.join(output_dir, fname)
    df.to_csv(out_path, index=False)
    print(f"✅ 保存 {out_path}")

print("🎯 指标计算完毕")
