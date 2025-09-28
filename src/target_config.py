# process_period_targets.py
from __future__ import annotations
import warnings
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.neighbors import LocalOutlierFactor

PERIOD_CONFIG: Dict[str, Dict] = {
    "1h": {"rolling_window": 48, "ewma": True, "lof_neighbors": 96, "annual_factor": (24*365) ** 0.5, "vj_W": 24, "vj_L": 96, "vj_tau": 1.6, "brk_N": 96, "brk_eps": 0.0015, "brk_vol": 1.2},
    "4h": {"rolling_window": 48, "ewma": True, "lof_neighbors": 48,  "annual_factor": (6*365) ** 0.5,  "vj_W": 12, "vj_L": 48, "vj_tau": 1.5, "brk_N": 48,  "brk_eps": 0.0015, "brk_vol": 1.2},
    "1d": {"rolling_window": 30, "ewma": True, "lof_neighbors": 30,  "annual_factor": (365) ** 0.5,     "vj_W": 7,  "vj_L": 30, "vj_tau": 1.4, "brk_N": 30,  "brk_eps": 0.001,  "brk_vol": 1.1},
}

def process_period_targets(df: pd.DataFrame, period: str, future_col: str, symbol_name: str | None = None) -> pd.DataFrame:
    assert period in PERIOD_CONFIG, f"Unknown period: {period}"
    cfg = PERIOD_CONFIG[period]

    feature_flags = {
        # 基础目标
        "target_logreturn": True,
        "target_binarytrend": True,
        "target_logsharpe": True,  # 这里作为“特征”生成 logsharpe_ratio，若需要可复制为 target
        # 波动率 & 结构
        "rolling_volatility": True,
        "parkinson_volatility": True,
        "ewma_volatility": cfg["ewma"],
        "return_skewness": True,
        "tail_bias": True,
        "amplitude_range": True,
        "atr_slope": True,
        # LOF
        "lof_detection": False,
        # 复合目标
        "target_multiagree": True,
        "target_trend_persistence": True,
        "target_pullback_prob": True,
        "target_sideway_detect": True,
        "target_breakout_count": True,
        "target_max_drawdown": True,
        "target_drawdown_prob": True,
        # 新增 4 个缺失目标
        "target_fundflow_strength": True,
        "target_vol_jump_prob": True,
        "target_realized_vol": True,
        "target_breakout_prob": True,
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
        df["ewmavolatility"] = df["logreturn"].ewm(span=max(2, win // 2), adjust=False).std().fillna(0)

    # === logSharpe: 作为“特征” ===
    if flags["target_logsharpe"]:
        vol = df.get("rolling_volatility", 1.0)
        df["target_logsharpe_ratio"] = df["logreturn"] / (vol + eps)
        # 如需 target，也可以：
        # df["target_logsharpe_ratio"] = df["logsharpe_ratio"]

    # === 结构 ===
    if flags["return_skewness"]:
        df["return_skewness"] = df["logreturn"].rolling(win).apply(lambda x: skew(x.dropna()), raw=False)
    if flags["tail_bias"]:
        # 高低影线差 / ATR
        df["tail_bias_norm"] = ((df["high"] - df[close]) - (df[close] - df["low"])) / (df["atr"] + eps)
    if flags["amplitude_range"]:
        df["amplitude_range"] = df["high"] - df["low"]
    if flags["atr_slope"]:
        df["atr_slope"] = df["atr"].diff()

    # === LOF 一次性批量（全是“特征”，不含 target） ===
    if flags["lof_detection"]:
        lof_cols = [c for c in [
            "logreturn", "rolling_volatility", "parkinson_volatility", "ewmavolatility",
            "return_skewness", "tail_bias_norm", "amplitude_range", "atr_slope",
        ] if c in df.columns and df[c].notna().all()]
        if lof_cols:
            neigh = min(max(PERIOD_CONFIG[period]["lof_neighbors"], int(len(df) * 0.1)), 50)
            lof = LocalOutlierFactor(n_neighbors=neigh, contamination=0.02)
            X = df[lof_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            scores = lof.fit_predict(X)
            df["lof_score_all"] = lof.negative_outlier_factor_
            df["is_outlier_all"] = (scores == -1).astype(int)

    # === 目标字段 ===
    if flags["target_logreturn"]:
        # 分位统计使用历史，避免泄露
        p = df["logreturn"].shift(1).rolling(win)
        p10, p90 = p.quantile(0.10), p.quantile(0.90)
        df["target_logreturn"] = df["logreturn"]
        df["target_return_outlier_lower_10"] = (df["logreturn"] < p10).astype(int)
        df["target_return_outlier_upper_90"] = (df["logreturn"] > p90).astype(int)
        df["logreturn_pct_rank"] = p.apply(lambda x: x.rank(pct=True).iloc[-1] if len(x) else np.nan, raw=False)

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
        # 连续方向长度（更稳）
        dir_ = np.sign(df["logreturn"].fillna(0))
        same_dir = dir_.shift(1).fillna(0) == dir_
        run = (~same_dir).cumsum()
        df["target_trend_persistence"] = same_dir.groupby(run).cumsum() + 1

    if flags["target_pullback_prob"]:
        # 方向反转且之前保持同向 → 回调
        dir_ = df["binary_trend"]
        rev = (dir_.shift(1) != dir_) & (dir_.shift(2) == dir_)
        df["target_pullback_prob"] = rev.astype(int)

    if flags["target_sideway_detect"]:
        # 阈值先保留常量，后续可换成分位阈值
        low_vol = df["rolling_volatility"].rolling(max(2, win // 2)).mean() < 0.02
        neutral = df["logreturn"].abs() < 0.001
        df["target_sideway_detect"] = (low_vol & neutral).astype(int)

    if flags["target_breakout_count"]:
        up_seq = df["logreturn"] > 0
        df["target_breakout_count"] = up_seq.groupby((~up_seq).cumsum()).cumsum()

    # === 新增目标：fundflow_strength（回归） ===
    if flags.get("target_fundflow_strength", False):
        # 需要以下列（已在融合阶段 shift/ffill，并可能做过标准化）：
        # exch_netflow(卖压取负)、stablecoin_mcap(增量)、etf_flow、cb_premium
        w_ex, w_st, w_etf, w_cb = 0.35, 0.25, 0.25, 0.15
        ex = df.get("exch_netflow")
        st = df.get("stablecoin_mcap")
        et = df.get("etf_flow")
        cb = df.get("cb_premium")
        # 差分/标准化（鲁棒）：
        def _z(x: pd.Series):
            if x is None: return None
            xm = x.astype(float)
            mu = xm.mean(); sd = xm.std(ddof=0)
            sd = sd if sd and sd > 1e-6 else 1.0
            return (xm - mu) / sd
        ex_z = -_z(ex) if ex is not None else 0.0   # 净流出为正卖压 → 取负
        st_d = st.diff() if st is not None else None
        st_z = _z(st_d) if st_d is not None else 0.0
        et_z = _z(et) if et is not None else 0.0
        cb_z = _z(cb) if cb is not None else 0.0
        df["target_fundflow_strength"] = (
            w_ex * (ex_z if isinstance(ex_z, pd.Series) else 0.0) +
            w_st * (st_z if isinstance(st_z, pd.Series) else 0.0) +
            w_etf * (et_z if isinstance(et_z, pd.Series) else 0.0) +
            w_cb * (cb_z if isinstance(cb_z, pd.Series) else 0.0)
        )

    # === 新增目标：vol_jump_prob（分类） ===
    if flags.get("target_vol_jump_prob", False):
        W = PERIOD_CONFIG[period]["vj_W"]; L = PERIOD_CONFIG[period]["vj_L"]; tau = PERIOD_CONFIG[period]["vj_tau"]
        hist_vol = df["rolling_volatility"].rolling(L).mean()
        fut = df["logreturn"].shift(-1).rolling(W).std()
        ratio = (fut / (hist_vol + 1e-6)).fillna(0.0)
        df["target_vol_jump_prob"] = (ratio >= tau).astype(int)

    # === 新增目标：realized_vol（回归） ===
    if flags.get("target_realized_vol", False):
        ann = PERIOD_CONFIG[period]["annual_factor"]
        # 未来 W 窗的实现波动（年化）：
        W = PERIOD_CONFIG[period]["vj_W"]
        rv = df["logreturn"].shift(-1).rolling(W).apply(lambda x: float(np.sqrt(np.sum(np.square(x.astype(float))) + 1e-12)), raw=False)
        df["target_realized_vol"] = rv * ann

    # === 新增目标：breakout_prob（分类） ===
    if flags.get("target_breakout_prob", False):
        N = PERIOD_CONFIG[period]["brk_N"]; eps = PERIOD_CONFIG[period]["brk_eps"]; vthr = PERIOD_CONFIG[period]["brk_vol"]
        # 简化口径：近 N 窗的局部高低 + 量能确认
        hi = df["high"].rolling(N, min_periods=2).max(); lo = df["low"].rolling(N, min_periods=2).min()
        vol = df.get("volume", pd.Series(np.nan, index=df.index)).astype(float)
        v_mean = vol.rolling(max(2, N//4)).mean(); v_ok = (vol / (v_mean + 1e-6)) >= vthr
        up_brk = (df["close"].shift(-1) >= (hi * (1.0 + eps))) & v_ok
        dn_brk = (df["close"].shift(-1) <= (lo * (1.0 - eps))) & v_ok
        df["target_breakout_prob"] = (up_brk | dn_brk).astype(int)

    # 清理临时列
    df.drop(columns=["logreturn", "binary_trend"], inplace=True, errors="ignore")
    return df
