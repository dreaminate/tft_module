# src/indicating.py
import os
import pandas as pd
import numpy as np
import warnings
from pandas.errors import PerformanceWarning

# 屏蔽 pandas 的性能告警，避免终端刷屏
warnings.filterwarnings("ignore", category=PerformanceWarning)
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
    # 新增扩展
    donchian_channel, keltner_channel, calculate_ppo, stoch_rsi, williams_r, tsi,
    ichimoku, psar, supertrend, heikin_ashi_trend,
    money_flow_index, chaikin_money_flow, price_volume_corr, pivot_point, linear_slope,
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
        # === 扩展开关 ===
        "regime_flags": True,
        "channels_squeeze": True,
        "hi_momentum": True,
        "ichimoku": True,
        "turning_psar_supertrend": True,
        "tail_risk": True,
        "candlestick_counts": True,
        "pv_imbalance": True,
        "cross_section": True,
        "pivot_hvn": True,
        "time_features": True,
        
    }

    # 3) 路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(project_root, "data", "crypto", timeframe)
    output_dir = os.path.join(project_root, "data", "crypto_indicated", timeframe)
    os.makedirs(output_dir, exist_ok=True)

    # 4) 滑动窗口（按周期）
    tf2win = {"1h": 48, "4h": 48, "1d": 48}
    win = tf2win.get(timeframe, 48)

    # 4.1 预计算 BTC 基准收益（按当前周期）
    btc_r_map = None
    btc_path = os.path.join(input_dir, f"BTC_USDT_{timeframe}_all.csv")
    if os.path.exists(btc_path):
        try:
            btc_df = pd.read_csv(btc_path, parse_dates=["datetime"])
            for col in ["open", "high", "low", "close", "volume"]:
                suffixed = f"{col}_BTC_USDT_{timeframe}"
                if suffixed in btc_df.columns:
                    btc_df.rename(columns={suffixed: col}, inplace=True)
            btc_df["log_close"] = np.log(btc_df["close"].replace(0, np.nan))
            btc_df["r_btc"] = btc_df["log_close"].diff()
            btc_r_map = dict(zip(btc_df["datetime"].values, btc_df["r_btc"].values))
        except Exception:
            btc_r_map = None

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

        # === A. Regime flags ===
        if enabled["regime_flags"]:
            # hist vol (realized) 近似：使用 close 的 rolling std（对数收益）
            r = np.log(df[close_col].replace(0, np.nan)).diff()
            std_20 = r.rolling(20, min_periods=10).std()
            ret_20 = r.rolling(20, min_periods=10).sum()
            std_1d = r.rolling(24 if timeframe=="1h" else (6 if timeframe=="4h" else 1), min_periods=1).std()

            df["trend_flag"] = ((df.get("adx", 0) > 25) & (ret_20.abs() > 2.0 * std_20)).astype(int)
            bw = None
            if {"boll_upper","boll_lower","ma20"}.issubset(df.columns):
                bw = boll_bandwidth(df)
                df["boll_bandwidth"] = df.get("boll_bandwidth", bw)
            atr_ratio = (df["atr"] / (df[close_col].abs() + 1e-9)) if "atr" in df.columns else pd.Series(0, index=df.index)
            df["range_flag"] = ((atr_ratio < 0.008) & (df.get("adx", 0) < 15) & ((df.get("boll_bandwidth", bw) if bw is not None else 0) < (df.get("boll_bandwidth", 0).rolling(500).quantile(0.3)))).astype(int)
            df["high_vol_flag"] = ((std_20 > std_20.rolling(100).quantile(0.7)) & (df.get("adx", 0) < 15)).astype(int)
            # stress 简化：极端负收益，若未来接入清算/资金费率可再增强
            df["stress_flag"] = ((r < -4.0 * (std_1d.replace(0, np.nan)))).astype(int)

        # === B. 通道 & 压缩 ===
        if enabled["channels_squeeze"]:
            dch, dcl = donchian_channel(df, window=20, high_col=high_col, low_col=low_col)
            df["donchian_high_20"] = dch
            df["donchian_low_20"] = dcl
            kc_mid, kc_up, kc_low = keltner_channel(df, period=20, multiplier=1.5, price_col=close_col, high_col=high_col, low_col=low_col)
            df["keltner_upper_20"] = kc_up
            df["keltner_lower_20"] = kc_low
            if {"boll_upper","boll_lower","ma20"}.issubset(df.columns):
                width_bb = boll_bandwidth(df)
                width_kc = (kc_up - kc_low) / (df["ma20"].abs() + 1e-9)
                perc = width_kc.rolling(500, min_periods=100).apply(lambda x: (x < x.iloc[-1]).mean())
                df["squeeze_on"] = ((width_bb < width_kc) & (perc < 0.3)).astype(int)

        # === C. 高阶动量/振荡 ===
        if enabled["hi_momentum"]:
            df["ppo"] = calculate_ppo(df, column=close_col)
            df["stoch_rsi"] = stoch_rsi(df, column=close_col)
            df["williams_r"] = williams_r(df, high_col=high_col, low_col=low_col, close_col=close_col)
            df["tsi"] = tsi(df, column=close_col)

        # === D. 一目均衡 ===
        if enabled["ichimoku"]:
            conv, base, span_a, span_b = ichimoku(df, high_col=high_col, low_col=low_col)
            df["ichimoku_conv"] = conv
            df["ichimoku_base"] = base
            df["ichimoku_span_a"] = span_a
            df["ichimoku_span_b"] = span_b
            df["cloud_thickness"] = (span_b - span_a)

        # === E. 转折 & 形态 ===
        if enabled["turning_psar_supertrend"]:
            ps, flip = psar(df, high_col=high_col, low_col=low_col)
            df["psar"] = ps
            df["psar_flip_flag"] = flip
            df["supertrend_10_3"] = supertrend(df, period=10, multiplier=3.0, high_col=high_col, low_col=low_col, close_col=close_col)
            df["heikin_ashi_trend"] = heikin_ashi_trend(df)

        # === F. Tail & Risk ===
        if enabled["tail_risk"]:
            rlog = np.log(df[close_col].replace(0, np.nan)).diff()
            win = 30 if timeframe != "1d" else 30
            df["ret_skew_30d"] = rlog.rolling(win, min_periods=10).skew()
            df["ret_kurt_30d"] = rlog.rolling(win, min_periods=10).kurt()
            # VaR / CVaR 粗略近似（历史分位）
            q = rlog.rolling(win, min_periods=10).quantile(0.05)
            df["var_95_30d"] = q
            # CVaR 近似：低于分位的平均
            def _cvar(x):
                if len(x.dropna()) == 0:
                    return np.nan
                qv = np.nanquantile(x, 0.05)
                y = x[x <= qv]
                return float(y.mean()) if len(y) else np.nan
            df["cvar_95_30d"] = rlog.rolling(win, min_periods=10).apply(_cvar, raw=False)
            df["z_score_gap"] = (df["open"] - df["close"].shift(1)) / (df.get("atr", 1.0) + 1e-9)

        # === G. 蜡烛型计数（简化版占比/计数） ===
        if enabled["candlestick_counts"]:
            body = (df["close"] - df["open"]).abs()
            range_ = (df["high"] - df["low"]).replace(0, np.nan)
            doji = (body / range_ < 0.1).astype(int)
            hammer = ((df["open"] > df["close"]) & ((df["open"] - df["low"]) / range_ > 0.6)).astype(int)
            engulf_up = ((df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["close"] >= df["open"].shift(1)) & (df["open"] <= df["close"].shift(1))).astype(int)
            df["doji_ratio_20"] = doji.rolling(20, min_periods=5).mean()
            df["hammer_cnt_20"] = hammer.rolling(20, min_periods=5).sum()
            df["engulfing_up_cnt_20"] = engulf_up.rolling(20, min_periods=5).sum()

        # === H. 量价失衡 ===
        if enabled["pv_imbalance"]:
            df["mfi"] = money_flow_index(df, high_col=high_col, low_col=low_col, close_col=close_col, volume_col=vol_col)
            df["cmf"] = chaikin_money_flow(df, high_col=high_col, low_col=low_col, close_col=close_col, volume_col=vol_col)
            df["price_volume_corr_20"] = price_volume_corr(df, window=20, close_col=close_col, volume_col=vol_col)

        # === I. 级差 / 横截面 ===
        if enabled["cross_section"]:
            df["ma200"] = calculate_ma(df, 200, column=close_col)
            df["price_position_ma200"] = (df[close_col] / (df["ma200"].replace(0, np.nan)) - 1.0)
            # beta_btc_60d：按当前周期使用等效60天窗口（1h=1440, 4h=360, 1d=60）与 BTC 的滚动 β（shift(1) 防泄露）
            if btc_r_map is not None and len(df) > 0:
                try:
                    df["log_close"] = np.log(df[close_col].replace(0, np.nan))
                    df["r_sym"] = df["log_close"].diff()
                    df["r_btc"] = pd.Series(df["datetime"]).map(pd.Series(btc_r_map))
                    bars_map = {"1h": 24*60, "4h": 6*60, "1d": 60}
                    roll = bars_map.get(timeframe, 60)
                    minp = max(roll // 2, 30)
                    m_rs = df["r_sym"].rolling(roll, min_periods=minp).mean()
                    m_rb = df["r_btc"].rolling(roll, min_periods=minp).mean()
                    cov = (df["r_sym"] * df["r_btc"]).rolling(roll, min_periods=minp).mean() - m_rs * m_rb
                    varb = (df["r_btc"] * df["r_btc"]).rolling(roll, min_periods=minp).mean() - (m_rb * m_rb)
                    df["beta_btc_60d"] = (cov / (varb.replace(0, np.nan))).shift(1)
                except Exception:
                    df["beta_btc_60d"] = np.nan

        # === J. Pivot & Volume Profile (HVN/LVN) ===
        if enabled["pivot_hvn"]:
            df["pivot_point"] = pivot_point(df, high_col=high_col, low_col=low_col, close_col=close_col)
            df["distance_to_pivot"] = (df[close_col] - df["pivot_point"]) / (df[close_col].abs() + 1e-9)
            # volume_profile_hvn/lvn：等效60天滚动成交量-价格直方图的高/低成交量节点（shift(1)）
            if len(df) > 0:
                try:
                    n = len(df)
                    hvn_vals = np.full(n, np.nan, dtype=float)
                    lvn_vals = np.full(n, np.nan, dtype=float)
                    bars_map = {"1h": 24*60, "4h": 6*60, "1d": 60}
                    bins_map = {"1h": 50, "4h": 40, "1d": 20}
                    roll = bars_map.get(timeframe, 60)
                    bins = bins_map.get(timeframe, 20)
                    minp = max(roll // 2, 30)
                    prices = df[close_col].to_numpy()
                    vols = df[vol_col].to_numpy()
                    for i in range(n):
                        start = max(0, i - roll + 1)
                        win_p = prices[start:i+1]
                        win_v = vols[start:i+1]
                        if len(win_p) < minp:
                            continue
                        lo = float(np.nanmin(win_p)); hi = float(np.nanmax(win_p))
                        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                            continue
                        hist, edges = np.histogram(win_p, bins=bins, range=(lo, hi), weights=win_v)
                        centers = (edges[:-1] + edges[1:]) / 2.0
                        if hist.size == 0:
                            continue
                        hvn_idx = int(np.nanargmax(hist))
                        hvn_center = float(centers[hvn_idx]) if np.isfinite(centers[hvn_idx]) else np.nan
                        pos_mask = hist > 0
                        if pos_mask.any():
                            # 在权重大于 0 的箱中选择最小者作为 LVN
                            masked = np.where(pos_mask, hist, np.nan)
                            lvn_idx = int(np.nanargmin(masked))
                            lvn_center = float(centers[lvn_idx]) if np.isfinite(centers[lvn_idx]) else np.nan
                        else:
                            lvn_center = np.nan
                        hvn_vals[i] = hvn_center
                        lvn_vals[i] = lvn_center
                    df["volume_profile_hvn"] = pd.Series(hvn_vals).shift(1)
                    df["volume_profile_lvn"] = pd.Series(lvn_vals).shift(1)
                except Exception:
                    df["volume_profile_hvn"] = np.nan
                    df["volume_profile_lvn"] = np.nan

        # === K. 时间特征（known_future） ===
        if enabled["time_features"] and "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
            hour = dt.dt.hour
            dow = dt.dt.dayofweek
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # === shift(1)：对重点新增列避免泄露 ===
        shift_cols = [
            # Regime flags
            "trend_flag","range_flag","high_vol_flag","stress_flag",
            # Channels & squeeze
            "squeeze_on",
            # Momentum/oscillator
            "ppo","stoch_rsi","williams_r","tsi",
            # Turning/patterns
            "psar","psar_flip_flag","supertrend_10_3","heikin_ashi_trend",
            # Tail & risk
            "ret_skew_30d","ret_kurt_30d","var_95_30d","cvar_95_30d","z_score_gap",
            # PV imbalance & cross-section
            "mfi","cmf","price_volume_corr_20","price_position_ma200","beta_btc_60d",
            # Pivot & time
            "pivot_point","distance_to_pivot","volume_profile_hvn","volume_profile_lvn","hour_sin","hour_cos","dow_sin","dow_cos",
        ]
        for c in shift_cols:
            if c in df.columns:
                df[c] = df[c].shift(1)

        # === ★ 派生：diff 与 slope_24h ===
        starred = [
            "trend_flag","range_flag",
            "ppo","stoch_rsi",
            "psar",
            "ret_skew_30d","ret_kurt_30d",
            "mfi",
            "price_position_ma200",
        ]
        slope_win = {"1h": 24, "4h": 6, "1d": 2}.get(timeframe, 24)
        for col in starred:
            if col in df.columns:
                try:
                    df[f"{col}_diff1"] = df[col].diff()
                    df[f"{col}_slope_{'24h'}"] = df[col].rolling(slope_win, min_periods=max(2, slope_win//2)).apply(linear_slope, raw=True)
                except Exception:
                    pass

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
                # === 新增扩展目标（★ 推荐派生） ===
                "trend_flag","range_flag","high_vol_flag","stress_flag",
                "donchian_high_20","donchian_low_20","keltner_upper_20","keltner_lower_20","squeeze_on",
                "ppo","stoch_rsi","williams_r","tsi",
                "ichimoku_conv","ichimoku_base","ichimoku_span_a","ichimoku_span_b","cloud_thickness",
                "psar","psar_flip_flag","supertrend_10_3","heikin_ashi_trend",
                "ret_skew_30d","ret_kurt_30d","var_95_30d","cvar_95_30d","z_score_gap",
                "engulfing_up_cnt_20","doji_ratio_20","hammer_cnt_20",
                "mfi","cmf","price_volume_corr_20",
                "price_position_ma200","beta_btc_60d","pivot_point","distance_to_pivot",
                "volume_profile_hvn","volume_profile_lvn",
                "hour_sin","hour_cos","dow_sin","dow_cos",
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