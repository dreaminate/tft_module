# target_config.py
"""
目标构造模块: compute_targets()
用于为每个周期 ( 1h / 4h / 1d )加密货币数据构造训练目标字段 ( target_ )和辅助特征 ( 非 target_ )。
✅ 分类目标: 
    - target_binarytrend: 上涨为 1, 非上涨为 0
    - target_multiagree: 多周期一致方向 ( 1 or 0 )
    - target_pullback_prob: 涨中跌、跌中涨的短期回调信号
    - target_sideway_detect: 当前是否为震荡区间 ( 低波动 + 中性涨幅 )
    - target_return_outlier_upper_75: logreturn 是否属于 P75 上分位
    - target_return_outlier_upper_90: logreturn 是否属于 P90 上分位
    - target_return_outlier_lower_25: logreturn 是否属于 P25 下分位
    - target_return_outlier_lower_10: logreturn 是否属于 P10 下分位
    - target_trend3class: 趋势三分类 ( 上涨 = 2 / 下跌 = 0 / 横盘 = 1 )

✅ 回归目标: 
    - target_logreturn: log(future / close)，对数收益率
    - target_logsharpe_ratio: logreturn / volatility, 风险调整收益
    - target_trend_persistence: 当前趋势持续步数 ( 如连续上涨 )
    - target_fundflow_strength: 链上资金流 ( 当前留空 )
    - target_breakout_count: 近期连续上涨或突破次数
    - target_max_drawdown: 历史回撤最大值 ( 近 N 步 )
    - target_drawdown_prob: 是否处于回撤状态 ( 近 N 步收益为负 )

✅ 波动率 / 辅助指标 ( 非 target ): 
    - rolling_volatility: log(1+std(logreturn))
    - parkinson_volatility: 基于高低价的波动率估计
    - ewma_volatility: 指数加权波动率
    - aparch_volatility / egarch_volatility: ARCH族波动率估计
    - vol_skewness: 波动率偏度
    - return_skewness: 收益率偏度
    - tail_bias: K线尾部偏度
    - amplitude_range: 振幅 ( high - low )
    - atr_slope: ATR 的斜率，趋势强弱
    - LOF 异常分数 + 异常标签 ( 每个字段 )

参数说明: 
    - df: 带有 OHLCV + future_close 的 DataFrame
    - period: 时间周期，如 "1h"
    - future_col: 未来价格列名 ( 一般为 "future_close" )
    - symbol_name: 如 "BTC_USDT"
    - feature_flags: 控制字段构造开关的字典
"""

import warnings
from arch import arch_model
from scipy.stats import skew
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd
from datetime import datetime

# === 1️⃣ 周期配置字典 ===
PERIOD_CONFIG = {
    "1h":  {"rolling_window": 48, "garch": False, "ewma": True,  "lof_neighbors": 96},
    "4h":  {"rolling_window": 48, "garch": False, "ewma": True,  "lof_neighbors": 48},
    "1d":  {"rolling_window": 30, "garch": False, "ewma": True ,"lof_neighbors": 30}
}

# === 2️⃣ 包装函数 ===



# 周期配置示例（需在主脚本或配置文件中定义）

def process_period_targets(df: pd.DataFrame, period: str, future_col: str, symbol_name: str = None):
    """
    执行目标构造、清洗 NaN，并打印清洗信息与删除区间。
    """
    assert period in PERIOD_CONFIG, f"[❌] 未知周期: {period}"
    cfg = PERIOD_CONFIG[period]

    feature_flags = {
        "target_logreturn": True,
        "target_binarytrend": True,
        "target_logsharpe": True,
        "rolling_volatility": True,
        "parkinson_volatility": True,
        "logsharpe_ratio": True,
        "ewma_volatility": cfg["ewma"],
        "garch_volatility": cfg["garch"],
        "vol_skewness": cfg["garch"],
        "return_skewness": True,
        "tail_bias": True,
        "amplitude_range": True,
        "atr_slope": True,
        "lof_detection": True,
        "fundflow_strength": False,
        "target_multiagree": True,
        "target_trend_persistence": True,
        "target_pullback_prob": True,
        "target_sideway_detect": True,
        "target_breakout_count": True,
        "target_max_drawdown": True,
        "target_drawdown_prob": True,
        "target_trend3class": True
    }

    df = df.copy()
    n_before = len(df)

    # 记录原始索引和时间
    df["_original_index"] = df.index
    df["_original_timestamp"] = df["timestamp"]

    # === 构造目标字段 ===
    df = compute_targets(df, period=period, future_col=future_col, feature_flags=feature_flags)

    # 找出 NaN 行
    nan_rows = df[df.isna().any(axis=1)]
    # === 打印 NaN 行详细信息 ===w
    if not nan_rows.empty:
        print(f"[⚠️] 发现 NaN 行，共 {len(nan_rows)} 行:")
        print(nan_rows.head())  # 打印 NaN 行的前几行进行详细查看
    # 删除 NaN 行
    df_cleaned = df.dropna(how="any").reset_index(drop=True)
    n_after = len(df_cleaned)

    print(f"[✅] 周期: {period} | 原始行数: {n_before} → 清洗后: {n_after} | 删除行数: {n_before - n_after}")

    # === 打印删除区间分析 ===
    if not nan_rows.empty:
        deleted_indices = nan_rows["_original_index"].values
        deleted_timestamps = nan_rows["_original_timestamp"].values

        is_continuous = (max(deleted_indices) - min(deleted_indices) + 1) == len(deleted_indices)

        if is_continuous:
            start_ts = int(deleted_timestamps[0])
            end_ts = int(deleted_timestamps[-1])
            start_dt = datetime.utcfromtimestamp(start_ts / 1000)
            end_dt = datetime.utcfromtimestamp(end_ts / 1000)
            print(f"[🧹] 删除行为连续: 时间范围 {start_dt} ~ {end_dt}")
        else:
            print(f"[🧹] 删除行为离散，共 {len(deleted_indices)} 行，前 10 行时间点：")
            for ts in deleted_timestamps[:10]:
                dt_str = datetime.utcfromtimestamp(int(ts) / 1000)
                print(f"   • {dt_str}")

    return df_cleaned


def compute_targets(df, period, future_col,feature_flags=None):
    warnings.filterwarnings("ignore")
    epsilon = 1e-6
    close_col = "close"

    if feature_flags is None:
        feature_flags = {
            "target_logreturn": True,
            "target_binarytrend": True,
            "target_logsharpe": True,
            "rolling_volatility": True,
            "parkinson_volatility": True,
            "logsharpe_ratio": True,
            "ewma_volatility": True,
            "return_skewness": True,
            "tail_bias": True,
            "amplitude_range": True,
            "atr_slope": True,
            "lof_detection": True,
            "fundflow_strength": False,
            "target_multiagree": True,
            "target_trend_persistence": True,
            "target_pullback_prob": True,
            "target_sideway_detect": True,
            "target_breakout_count": True,
            "target_max_drawdown": True,
            "target_drawdown_prob": True,
            "target_trend3class": True
        }
           

    df["logreturn"] = np.log(df[future_col] / df[close_col])
    df["binary_trend"] = (df["logreturn"] > 0).astype(int)

    
    if feature_flags["rolling_volatility"]:
        df["rolling_volatility"] = np.log1p(df["logreturn"].rolling(48).std().fillna(0))

    if feature_flags["parkinson_volatility"]:
        df["parkinson_volatility"] = np.log1p(((np.log(df["high"] / df["low"])) ** 2).rolling(48).mean().fillna(0))

    if feature_flags["logsharpe_ratio"]:
        df["logsharpe_ratio"] = df["logreturn"] / (df.get("rolling_volatility", 1.0) + epsilon)

    if feature_flags["ewma_volatility"] and period in ["1h", "4h","1d"]:
        print(f"[🌀] 正在计算 ewma_volatility - {period}")
        df["ewmavolatility"] = df["logreturn"].ewm(span=24, adjust=False).std().fillna(0)

   

    if feature_flags["return_skewness"]:
        df["return_skewness"] = df["logreturn"].rolling(7).apply(lambda x: skew(x.dropna()), raw=False)

    if feature_flags["tail_bias"]:
        df["tail_bias"] = (df["high"] - df["close"]) - (df["close"] - df["low"])

    if feature_flags["amplitude_range"]:
        df["amplitude_range"] = df["high"] - df["low"]

    if feature_flags["atr_slope"]:
        df["atr_slope"] = df["atr"].diff()

    if feature_flags["lof_detection"]:
        lof_fields = [
            "logreturn", 
            "rolling_volatility", "parkinson_volatility",
            "logsharpe_ratio", "return_skewness", "vol_skewness",
            "tail_bias", "amplitude_range", "atr_slope"
                    ]

        if feature_flags["ewma_volatility"] and period in ["1h", "4h", "1d"]:
            lof_fields.append("ewmavolatility")

        available = [col for col in lof_fields if col in df.columns and df[col].isna().sum() == 0]
        if available:
            default_n_neighbors = {"1h": 96, "4h": 48, "1d": 30}[period]
            for feat in available:
                feat_data = df[[feat]].dropna()
                if len(feat_data) < default_n_neighbors + 1:
                    continue
                dynamic_neighbors = min(max(default_n_neighbors, int(len(feat_data) * 0.1)), 50)
                try:
                    lof = LocalOutlierFactor(n_neighbors=dynamic_neighbors, contamination=0.02)
                    scores = lof.fit_predict(feat_data)
                    df.loc[feat_data.index, f"lof_score_{feat}"] = lof.negative_outlier_factor_
                    df.loc[feat_data.index, f"is_outlier_{feat}"] = (scores == -1).astype(int)
                except Exception:
                    df[f"lof_score_{feat}"] = np.nan
                    df[f"is_outlier_{feat}"] = 0

    # 目标构造
    if feature_flags["target_logreturn"]:
        df["target_logreturn"] = df["logreturn"]
    if feature_flags.get("target_logreturn"):
        rolling_window_q = {"1h": 96, "4h": 48, "1d": 30}.get(period, 96)
        p25 = df["logreturn"].rolling(rolling_window_q).quantile(0.25)
        p75 = df["logreturn"].rolling(rolling_window_q).quantile(0.75)
        p10 = df["logreturn"].rolling(rolling_window_q).quantile(0.10)
        p90 = df["logreturn"].rolling(rolling_window_q).quantile(0.90)

        df["target_return_outlier_upper_75"] = (df["logreturn"] > p75).astype(int)
        df["target_return_outlier_upper_90"] = (df["logreturn"] > p90).astype(int)
        df["target_return_outlier_lower_25"] = (df["logreturn"] < p25).astype(int)
        df["target_return_outlier_lower_10"] = (df["logreturn"] < p10).astype(int)
        df["logreturn_pct_rank"] = (
        df["logreturn"].rolling(rolling_window_q)
        .apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    )
    if feature_flags.get("target_logreturn"):
        rolling_window_q = {"1h": 96, "4h": 48, "1d": 30}.get(period, 96)
        p25 = df["logreturn"].rolling(rolling_window_q).quantile(0.25)
        p75 = df["logreturn"].rolling(rolling_window_q).quantile(0.75)
        p10 = df["logreturn"].rolling(rolling_window_q).quantile(0.10)
        p90 = df["logreturn"].rolling(rolling_window_q).quantile(0.90)

        df["target_return_outlier_upper_75"] = (df["logreturn"] > p75).astype(int)
        df["target_return_outlier_upper_90"] = (df["logreturn"] > p90).astype(int)
        df["target_return_outlier_lower_25"] = (df["logreturn"] < p25).astype(int)
        df["target_return_outlier_lower_10"] = (df["logreturn"] < p10).astype(int)
        df["logreturn_pct_rank"] = (
            df["logreturn"].rolling(rolling_window_q)
            .apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
        )

    
    

    if feature_flags.get("target_max_drawdown"):
        window_dd = {"1h": 96, "4h": 48, "1d": 30}.get(period, 96)
        cum_return = df["logreturn"].cumsum()
        rolling_max = cum_return.rolling(window_dd, min_periods=1).max()
        df["target_max_drawdown"] = (cum_return - rolling_max).fillna(0)

    if feature_flags.get("target_drawdown_prob"):
        window_dd = {"1h": 96, "4h": 48, "1d": 30}.get(period, 96)
        cum_return = df["logreturn"].cumsum()
        rolling_max = cum_return.rolling(window_dd, min_periods=1).max()
        df["target_drawdown_prob"] = (cum_return < rolling_max).astype(int)
        if feature_flags["target_binarytrend"]:
            df["target_binarytrend"] = df["binary_trend"]

    if feature_flags["target_logsharpe"] and "logsharpe_ratio" in df:
        df["target_logsharpe_ratio"] = df["logsharpe_ratio"]

    if feature_flags["target_multiagree"]:
        df["target_multiagree"] = df["binary_trend"].rolling(3, min_periods=1).apply(lambda x: int(len(set(x)) == 1))

    if feature_flags["target_trend_persistence"]:
        persistence = (df["binary_trend"] == df["binary_trend"].shift()).astype(int)
        df["target_trend_persistence"] = persistence.groupby((persistence != 1).cumsum()).cumsum()

    if feature_flags["target_pullback_prob"]:
        direction = df["binary_trend"]
        reversal = (direction.shift(1) != direction) & (direction.shift(2) == direction)
        df["target_pullback_prob"] = reversal.astype(int)

    if feature_flags["target_sideway_detect"]:
        low_vol = df["rolling_volatility"].rolling(24).mean() < 0.02
        neutral_ret = df["logreturn"].abs() < 0.001
        df["target_sideway_detect"] = (low_vol & neutral_ret).astype(int)

    if feature_flags["fundflow_strength"]:
        df["fundflow_strength"] = np.nan  # placeholder for future on-chain data

    if feature_flags["target_breakout_count"]:
        up = df["logreturn"] > 0
        df["target_breakout_count"] = up.groupby((~up).cumsum()).cumsum()

    df.drop(columns=["logreturn"], inplace=True, errors="ignore")
    return df   

