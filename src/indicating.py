# src/indicating.py
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# === 自研指标库 ===
from indicators import (
    calculate_ma, calculate_rsi, calculate_macd, calculate_kdj,
    calculate_momentum, calculate_vol_ma, calculate_vol_change,
    calculate_obv, calculate_bollinger, calculate_atr,
    calculate_cci, calculate_adx, calculate_vwap,
    macd_bullish_cross, rsi_overbought, rsi_oversold, price_above_ma,
    breakout_recent_high, pullback_from_high, trendline_slope, ma_diff,
    volume_relative, volume_percentile, support_level, resistance_level,
    distance_to_support, distance_to_resistance,
    volume_spike_near_resist, volume_spike_near_support,
    obv_diff, obv_pct_change, obv_slope,
    atr_ratio_price, atr_ratio_range, atr_change, range_to_atr,
    boll_pctb, boll_bandwidth,
)

# 新增：局部归一化工具
from groupwise_rolling_norm import groupwise_rolling_norm
def indicating_main(timeframe="1h"):
    # 1) 周期
    timeframe = timeframe  # 可用 CLI 覆盖
    # 2) 指标/流程开关
    enabled = {
        "ma": True, "rsi": True, "macd": True, "kdj": True,
        "momentum": True, "vol_ma": True, "vol_change": True, "obv": True,
        "boll": True, "atr": True, "cci": True, "adx": True, "vwap": True,
        "signal_macd_cross": True, "signal_rsi_ob_os": True,
        "signal_price_above_ma": True, "signal_breakout_pullback": True,
        "trend_strength": True, "volume_features": True, "support_resistance": True,
        "obv_feats": True,
        "atr_feats": True,
        "boll_feats": True,
        "local_norm": True,   # 新增：是否做局部滑动归一化
        "lof": False,          # 是否做 LOF
        
    }

    # 3) 路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(project_root, "data", "crypto", timeframe)
    output_dir = os.path.join(project_root, "data", "crypto_indicated", timeframe)
    os.makedirs(output_dir, exist_ok=True)

    # 4) 滑动窗口（按周期）
    tf2win = {"1h": 48, "4h": 48, "1d": 48}
    win = tf2win.get(timeframe, 48)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(input_dir, fname), parse_dates=["datetime"])
        symbol_name = fname.replace(f"_{timeframe}_all.csv", "")
        df["symbol"] = symbol_name
        df["period"] = timeframe  # 为分组归一化准备

        print(f"📄 {fname}: {timeframe}")

        # 如果原始字段带后缀 → 去掉，统一字段名
        for col in ["open", "high", "low", "close", "volume"]:
            suffixed = f"{col}_{symbol_name}_{timeframe}"
            if suffixed in df.columns:
                df.rename(columns={suffixed: col}, inplace=True)

        close_col, high_col, low_col, vol_col = "close", "high", "low", "volume"

        # === 技术指标 ===
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
        
        if enabled.get("adx", False):
            adx, plus_di, minus_di = calculate_adx(df, period=14, high_col="high", low_col="low", close_col="close")
            df["adx"] = adx
            df["plus_di"] = plus_di
            df["minus_di"] = minus_di
            df["adx_scaled"] = df["adx"] / 100.0
        if enabled["vwap"]:
            df["vwap"] = calculate_vwap(df, column_price=close_col, column_volume=vol_col, high_col=high_col, low_col=low_col)

        # === 交易信号 ===
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

        # === 趋势/量能/支撑阻力 ===
        if enabled["trend_strength"] and {"close"}.issubset(df.columns):
            df["trendline_slope_20"] = trendline_slope(df)
            if {"ma5", "ma20"}.issubset(df.columns):
                df["ma_diff_5_20"] = ma_diff(df)
        if enabled["volume_features"]:
            df["volume_relative_20"] = volume_relative(df)
            df["volume_percentile"] = volume_percentile(df)
        if enabled["support_resistance"]:
            df["support_level_20"] = support_level(df, low_col=low_col)
            df["resistance_level_20"] = resistance_level(df, high_col=high_col)
            df["distance_to_support"] = distance_to_support(df)
            df["distance_to_resistance"] = distance_to_resistance(df)
            if {"volume_relative_20", "distance_to_support", "distance_to_resistance"}.issubset(df.columns):
                df["volume_spike_near_support"] = volume_spike_near_support(df)
                df["volume_spike_near_resist"] = volume_spike_near_resist(df)
            # === OBV 强化特征 ===
        if enabled["obv_feats"] and "obv" in df:
            df["obv_diff1"] = obv_diff(df)
            df["obv_pct"] = obv_pct_change(df)
            df["obv_slope_14"] = obv_slope(df, w=14)

        # === ATR 强化特征 ===
        if enabled["atr_feats"] and "atr" in df:
            df["atr_ratio_price"] = atr_ratio_price(df)
            df["atr_ratio_range"] = atr_ratio_range(df)
            df["atr_change"] = atr_change(df)
            df["range_to_atr"] = range_to_atr(df)
            # 额外：给局部归一化用的 atr_slope（不泄露）
            df["atr_slope"] = df["atr"].diff()

        # === 布林带强化特征 ===
        if enabled["boll_feats"] and {"boll_upper","boll_lower","ma20","close"}.issubset(df.columns):
            df["boll_pctb"] = boll_pctb(df)
            df["boll_bandwidth"] = boll_bandwidth(df)
        
        # 补充：通用振幅列（norm_targets 里用到了）
        df["amplitude_range"] = (df["high"] - df["low"])
        # === 局部滑动归一化（不覆盖原始列） ===
        if enabled.get("local_norm", True):
            norm_targets = [
                # 价格/均线/布林/支撑阻力
                "open","high","low","close","ma5","ma20","boll_upper","boll_lower",
                "support_level_20","resistance_level_20","distance_to_support","distance_to_resistance",
                # 量能
                "volume","vol_ma","volume_relative_20","obv", 
                # 波动率/幅度/斜率/偏度
                "atr","amplitude_range","atr_slope",
                # 你之后会并入的：rolling_volatility/parkinson/ewmavol 等
                # 如果不在原始指标里，这里没有也没关系
                "trendline_slope_20","ma_diff_5_20",
                # ✅ 新增：震荡 / 趋势 强度
                "cci",              # CCI 无界，z-score 很重要
                "adx_scaled",       # 用缩放后的 ADX（0~1）
                # ✅ 新增：成交量加权均价（不同币种价格量纲差异大）
                "vwap",
                # ✅ 新增：OBV 衍生
                "obv_diff1","obv_pct","obv_slope_14",
            ]
            exist_cols = [c for c in norm_targets if c in df.columns]
            if exist_cols:
                df = groupwise_rolling_norm(
                        df,
                        cols=exist_cols,
                        group_cols=["symbol", "period"],
                        window=win,
                        methods=["z", "mm"],
                        suffix_policy="with_window",
                        min_periods=win,       # 默认窗口1/4
                        eps=1e-7,               # 防止除 0
                        clip_z=5.0,             # Z-score 裁剪 [-5, 5]
                        clip_mm=(0.0, 1.0),     # Min-Max 裁剪 [0, 1]
                    )

                # 1) 统计归一化后的列（只清这些列的 NaN）
                zn_cols = [c for c in df.columns if c.endswith(f"_zn{win}")]
                mm_cols = [c for c in df.columns if c.endswith(f"_mm{win}")]
                normed_cols = zn_cols + mm_cols

                # 2) 按组裁掉 warm-up（确保 rolling 统计完整）
                df = df.sort_values(["symbol", "period", "datetime"])
                df["_row_id"] = df.groupby(["symbol", "period"]).cumcount()
                before = len(df)
                df = df[df["_row_id"] >= win].drop(columns="_row_id")
                after = len(df)
                print(f"🧹 裁掉 Warm-up 行: {before - after} 条（窗口={win}）")

                # 3) 再把仍然含有 NaN 的行去掉（只检查归一化后的列）
                if normed_cols:
                    before = len(df)
                    df = df.dropna(subset=normed_cols, how="any")
                    after = len(df)
                    print(f"🧽 归一化列 NaN 清理: 删除 {before - after} 行（检查 {len(normed_cols)} 列）")

                # 4) 可选：限制极端值
                for c in zn_cols:
                    df[c] = df[c].clip(-5, 5)


        # === LOF（优先使用 *_zn{win} 列作为输入） ===
        if enabled["lof"]:
            lof_candidates = [
                "rsi","macd_hist","kdj_j","volume","boll_upper","boll_lower","atr","vol_change",
                "pullback_from_high","volume_relative_20","distance_to_support","distance_to_resistance",
            ]
            default_n_neighbors = {"1h": 24, "4h": 12, "1d": 6}.get(timeframe, 24)
            lof_contamination = 0.01

            for base in lof_candidates:
                src = f"{base}_zn{win}" if f"{base}_zn{win}" in df.columns else base
                if src not in df.columns:
                    continue
                feat_data = df[[src]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(feat_data) < default_n_neighbors + 1:
                    print(f"⚠️ 跳过 {src}：样本不足（{len(feat_data)} 条）")
                    continue
                dynamic_neighbors = max(default_n_neighbors, int(len(feat_data) * 0.1))
                try:
                    lof = LocalOutlierFactor(n_neighbors=dynamic_neighbors, contamination=lof_contamination)
                    scores = lof.fit_predict(feat_data)
                    df.loc[feat_data.index, f"lof_score_{base}"] = lof.negative_outlier_factor_
                    df.loc[feat_data.index, f"is_outlier_{base}"] = (scores == -1).astype(int)
                except Exception as e:
                    print(f"❌ LOF 错误：{src} → {e}")
                    df[f"lof_score_{base}"] = np.nan
                    df[f"is_outlier_{base}"] = 0

            # 多字段联合 LOF（可选）
            joint_cols = []
            for base in lof_candidates:
                if f"{base}_zn{win}" in df.columns:
                    joint_cols.append(f"{base}_zn{win}")
                elif base in df.columns:
                    joint_cols.append(base)
            joint_cols = [c for c in joint_cols if c in df.columns]
            if len(joint_cols) >= 2:
                X = df[joint_cols].replace([np.inf, -np.inf], np.nan).dropna()
                idx = X.index
                if len(idx) >= default_n_neighbors + 1:
                    n_neighbors = max(default_n_neighbors, int(len(idx) * 0.1))
                    lof_joint = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.02)
                    yhat = lof_joint.fit_predict(X)
                    df.loc[idx, "lof_score_all"] = lof_joint.negative_outlier_factor_
                    df.loc[idx, "is_outlier_all"] = (yhat == -1).astype(int)

        # === 保存
        out_path = os.path.join(output_dir, fname)
        df.to_csv(out_path, index=False)
        print(f"✅ 保存 {out_path}")

    print("🎯 指标计算完毕")
# python src/indicating.py