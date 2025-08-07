"""
目标构造模块（精简版）
--------------------------------------------------
* 删除重复的 logreturn 分位计算代码
* `tail_bias` 归一化到 ATR 尺度 (`tail_bias_norm`)
* LOF 仅一次 fit，多字段同时评分，速度更快
* 保留原先 14 个 target，不生成 `target_trend3class`
--------------------------------------------------
调用:
    df = process_period_targets(df, period="1h", future_col="future_close")
"""

from __future__ import annotations
import warnings
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.neighbors import LocalOutlierFactor

# ------------------------------------------------------------------
# 周期参数
# ------------------------------------------------------------------
PERIOD_CONFIG: Dict[str, Dict] = {
    "1h": {"rolling_window": 48, "ewma": True, "lof_neighbors": 96},
    "4h": {"rolling_window": 48, "ewma": True, "lof_neighbors": 48},
    "1d": {"rolling_window": 30, "ewma": True, "lof_neighbors": 30},
}

# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------

def process_period_targets(df: pd.DataFrame, period: str, future_col: str, symbol_name: str | None = None) -> pd.DataFrame:
    assert period in PERIOD_CONFIG, f"Unknown period: {period}"
    cfg = PERIOD_CONFIG[period]

    feature_flags = {
        # 基础目标
        "target_logreturn": True,
        "target_binarytrend": True,
        "target_logsharpe": True,
        # 波动率 & 结构
        "rolling_volatility": True,
        "parkinson_volatility": True,
        "ewma_volatility": cfg["ewma"],
        "return_skewness": True,
        "tail_bias": True,
        "amplitude_range": True,
        "atr_slope": True,
        # LOF
        "lof_detection": True,
        # 复合目标
        "target_multiagree": True,
        "target_trend_persistence": True,
        "target_pullback_prob": True,
        "target_sideway_detect": True,
        "target_breakout_count": True,
        "target_max_drawdown": True,
        "target_drawdown_prob": True,
        # trend3class 留空
    }

    df = df.copy()
    df["_orig_idx"] = df.index
    df["_orig_ts"] = df["timestamp"]

    df = compute_targets(df, period, future_col, feature_flags)

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)
    print(f"[✅] {period} | rows: {n_before} → {n_after} (drop {n_before-n_after})")
    return df

# ------------------------------------------------------------------
# 计算目标 & 辅助特征
# ------------------------------------------------------------------

def compute_targets(df: pd.DataFrame, period: str, future_col: str, flags: Dict[str, bool]) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    eps = 1e-6
    close = "close"

    # === 基础 ===
    df["logreturn"] = np.log(df[future_col] / df[close])
    df["binary_trend"] = (df["logreturn"] > 0).astype(int)

    win = PERIOD_CONFIG[period]["rolling_window"]

    # === 波动率 ===
    if flags["rolling_volatility"]:
        df["rolling_volatility"] = np.log1p(df["logreturn"].rolling(win).std().fillna(0))
    if flags["parkinson_volatility"]:
        df["parkinson_volatility"] = np.log1p(((np.log(df["high"] / df["low"])) ** 2).rolling(win).mean().fillna(0))
    if flags["ewma_volatility"]:
        df["ewmavolatility"] = df["logreturn"].ewm(span=win//2, adjust=False).std().fillna(0)

    # logSharpe
    if flags["target_logsharpe"]:
        vol = df.get("rolling_volatility", 1.0)
        df["target_logsharpe_ratio"] = df["logreturn"] / (vol + eps)

    # 结构
    if flags["return_skewness"]:
        df["return_skewness"] = df["logreturn"].rolling(win).apply(lambda x: skew(x.dropna()), raw=False)
    if flags["tail_bias"]:
        df["tail_bias_norm"] = ((df["high"] - df[close]) - (df[close] - df["low"])) / (df["atr"] + eps)
    if flags["amplitude_range"]:
        df["amplitude_range"] = df["high"] - df["low"]
    if flags["atr_slope"]:
        df["atr_slope"] = df["atr"].diff()

    # === LOF 一次性批量 ===
    if flags["lof_detection"]:
        lof_cols = [c for c in [
            "logreturn", "rolling_volatility", "parkinson_volatility", "ewmavolatility",
            "logsharpe_ratio", "return_skewness", "tail_bias_norm", "amplitude_range", "atr_slope",
        ] if c in df.columns and df[c].notna().all()]
        if lof_cols:
            neigh = min(max(PERIOD_CONFIG[period]["lof_neighbors"], int(len(df) * 0.1)), 50)
            lof = LocalOutlierFactor(n_neighbors=neigh, contamination=0.02)
            scores = lof.fit_predict(df[lof_cols])
            df["lof_score_all"] = lof.negative_outlier_factor_
            df["is_outlier_all"] = (scores == -1).astype(int)

        # === 目标字段 ===
    if flags["target_logreturn"]:
        df["target_logreturn"] = df["logreturn"]

        # *仅* 极端 10% / 90% 分位异常
        p = df["logreturn"].rolling(win)
        p10, p90 = p.quantile(0.10), p.quantile(0.90)
        df["target_return_outlier_lower_10"] = (df["logreturn"] < p10).astype(int)
        df["target_return_outlier_upper_90"] = (df["logreturn"] > p90).astype(int)
        df["logreturn_pct_rank"] = p.apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)

    if flags["target_binarytrend"]:
        df["target_binarytrend"] = df["binary_trend"]

    if flags["target_max_drawdown"] or flags["target_drawdown_prob"]:
        cum_ret = df["logreturn"].cumsum()
        roll_max = cum_ret.rolling(win, min_periods=1).max()
        dd = cum_ret - roll_max
        if flags["target_max_drawdown"]:
            df["target_max_drawdown"] = dd
        if flags["target_drawdown_prob"]:
            df["target_drawdown_prob"] = (dd < 0).astype(int)

    if flags["target_trend_persistence"]:
        pers = (df["binary_trend"] == df["binary_trend"].shift()).astype(int)
        df["target_trend_persistence"] = pers.groupby((pers != 1).cumsum()).cumsum()

    if flags["target_pullback_prob"]:
        dir_ = df["binary_trend"]
        rev = (dir_.shift(1) != dir_) & (dir_.shift(2) == dir_)
        df["target_pullback_prob"] = rev.astype(int)

    if flags["target_sideway_detect"]:
        low_vol = df["rolling_volatility"].rolling(win//2).mean() < 0.02
        neutral = df["logreturn"].abs() < 0.001
        df["target_sideway_detect"] = (low_vol & neutral).astype(int)

    if flags["target_breakout_count"]:
        up_seq = df["logreturn"] > 0
        df["target_breakout_count"] = up_seq.groupby((~up_seq).cumsum()).cumsum()

    # 清理临时列
    df.drop(columns=["logreturn", "binary_trend"], inplace=True, errors="ignore")
    return df
        
