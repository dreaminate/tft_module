import pandas as pd
import numpy as np

"""indicators_cleaned.py

统一技术指标、信号特征以及关键水位函数。
所有函数均：
- 仅依赖 Pandas / NumPy；
- 遇到 inf / NaN 统一用 `_safe` 清洗；
- 不修改传入 df（无副作用）；
- 返回 `pd.Series`，与 df 索引对齐。

默认窗口、阈值可通过 kwargs 覆盖。
"""

# =========================
# 全局默认参数
# =========================
DEFAULTS = {
    "ma_window_short": 5,
    "ma_window_mid": 20,
    "rsi_period": 14,
    "macd_short": 12,
    "macd_long": 26,
    "macd_signal": 9,
    "kdj_period": 9,
    "momentum_period": 5,
    "atr_period": 14,
    "boll_period": 20,
    "cci_period": 20,
    "adx_period": 14,
    "support_resist_window": 20,
    "trendline_window": 20,
    "volume_rel_window": 20,
    "volume_pct_window": 120,
    "volume_spike_thresh": 1.5,
    "distance_thresh": 0.02,
}


# =========================
# 工具函数
# =========================

def _safe(series: pd.Series, fill: float | int = 0):
    """Replace inf/NaN with given fill value."""
    return series.replace([np.inf, -np.inf], np.nan).fillna(fill)


# =========================
# 技术指标系列
# =========================

def calculate_ma(df: pd.DataFrame, window: int = DEFAULTS["ma_window_short"], column: str = "close") -> pd.Series:
    return _safe(df[column].rolling(window=window).mean())


def calculate_rsi(df: pd.DataFrame, period: int = DEFAULTS["rsi_period"], column: str = "close") -> pd.Series:
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))
    return _safe(rsi)


def calculate_macd(
    df: pd.DataFrame,
    short_window: int = DEFAULTS["macd_short"],
    long_window: int = DEFAULTS["macd_long"],
    signal_window: int = DEFAULTS["macd_signal"],
    column: str = "close",
):
    ema_short = df[column].ewm(span=short_window, adjust=False).mean()
    ema_long = df[column].ewm(span=long_window, adjust=False).mean()
    macd_line = _safe(ema_short - ema_long)
    macd_signal = _safe(macd_line.ewm(span=signal_window, adjust=False).mean())
    macd_hist = _safe(macd_line - macd_signal)
    return macd_line, macd_signal, macd_hist


def calculate_kdj(
    df: pd.DataFrame,
    period: int = DEFAULTS["kdj_period"],
    column: str = "close",
    high_col: str = "high",
    low_col: str = "low",
):
    low_min = df[low_col].rolling(window=period, min_periods=1).min()
    high_max = df[high_col].rolling(window=period, min_periods=1).max()
    rsv = (df[column] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    return _safe(k), _safe(d), _safe(j)


def calculate_momentum(df: pd.DataFrame, period: int = DEFAULTS["momentum_period"], column: str = "close") -> pd.Series:
    return _safe(df[column] - df[column].shift(period))


def calculate_vol_ma(df: pd.DataFrame, window: int = DEFAULTS["ma_window_short"], column: str = "volume") -> pd.Series:
    return _safe(df[column].rolling(window=window).mean())


def calculate_vol_change(df: pd.DataFrame, column: str = "volume", clip_limit: float = 5.0) -> pd.Series:
    change = df[column].pct_change().clip(lower=-clip_limit, upper=clip_limit)
    return _safe(change)


def calculate_obv(df: pd.DataFrame, column_price: str = "close", column_volume: str = "volume") -> pd.Series:
    direction = np.sign(df[column_price].diff()).fillna(0)
    obv = (direction * df[column_volume]).cumsum()
    return _safe(obv)


def calculate_bollinger(df: pd.DataFrame, period: int = DEFAULTS["boll_period"], column: str = "close"):
    ma = _safe(df[column].rolling(window=period).mean())
    std = df[column].rolling(window=period).std()
    upper = _safe(ma + 2 * std)
    lower = _safe(ma - 2 * std)
    return ma, upper, lower


def calculate_atr(
    df: pd.DataFrame,
    period: int = DEFAULTS["atr_period"],
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    tr = pd.concat(
        [
            df[high_col] - df[low_col],
            (df[high_col] - df[close_col].shift()).abs(),
            (df[low_col] - df[close_col].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return _safe(atr)


def calculate_cci(
    df: pd.DataFrame,
    period: int = DEFAULTS["cci_period"],
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.Series:
    tp = (df[high_col] + df[low_col] + df[close_col]) / 3
    sma = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-10)
    return _safe(cci)


def calculate_adx(
    df: pd.DataFrame,
    period: int = DEFAULTS["adx_period"],
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
):
    up_move = df[high_col].diff()
    down_move = df[low_col].shift() - df[low_col]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            df[high_col] - df[low_col],
            (df[high_col] - df[close_col].shift()).abs(),
            (df[low_col] - df[close_col].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period).mean() / (atr + 1e-10)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    adx = _safe(pd.Series(dx).ewm(alpha=1 / period).mean())
    return adx, _safe(plus_di), _safe(minus_di)


def calculate_vwap(
    df: pd.DataFrame,
    period: int | None = None,
    column_price: str = "close",
    column_volume: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
):
    """Return VWAP series (rolling if period provided)."""
    tp = (df[high_col] + df[low_col] + df[column_price]) / 3
    if period is None:
        pv = (tp * df[column_volume]).cumsum()
        v = df[column_volume].cumsum()
        return _safe(pv / v)
    pv = (tp * df[column_volume]).rolling(window=period, min_periods=1).sum()
    v = df[column_volume].rolling(window=period, min_periods=1).sum()
    return _safe(pv / v)


# =========================
# 买卖信号
# =========================

def macd_bullish_cross(df: pd.DataFrame, col_hist: str = "macd_hist") -> pd.Series:
    return ((df[col_hist] > 0) & (df[col_hist].shift(1) <= 0)).astype(int)


def rsi_overbought(df: pd.DataFrame, col: str = "rsi", thresh: float = 70) -> pd.Series:
    return (df[col] > thresh).astype(int)


def rsi_oversold(df: pd.DataFrame, col: str = "rsi", thresh: float = 30) -> pd.Series:
    return (df[col] < thresh).astype(int)


def price_above_ma(df: pd.DataFrame, price_col: str = "close", ma_col: str = "ma20") -> pd.Series:
    return (df[price_col] > df[ma_col]).astype(int)


def breakout_recent_high(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    window: int = DEFAULTS["support_resist_window"],
) -> pd.Series:
    rolling_high = df[high_col].rolling(window, min_periods=window).max()
    return (df[price_col] > rolling_high.shift(1)).astype(int)


def pullback_from_high(
    df: pd.DataFrame,
    price_col: str = "close",
    high_col: str = "high",
    window: int = DEFAULTS["support_resist_window"],
) -> pd.Series:
    rolling_high = df[high_col].rolling(window, min_periods=window).max()
    return _safe((rolling_high - df[price_col]) / rolling_high.clip(lower=1e-9))


# =========================
# 趋势强化
# =========================

def trendline_slope(df: pd.DataFrame, price_col: str = "close", window: int = DEFAULTS["trendline_window"]) -> pd.Series:
    def _slope(arr):
        n = len(arr)
        x = np.arange(n)
        x_mean = x.mean()
        y_mean = arr.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        return (((x - x_mean) * (arr - y_mean)).sum()) / denom

    return _safe(
        df[price_col]
        .rolling(window, min_periods=window)
        .apply(_slope, raw=True)
    )


def ma_diff(df: pd.DataFrame, short_ma: str = "ma5", long_ma: str = "ma20") -> pd.Series:
    return _safe(df[short_ma] - df[long_ma])


# =========================
# 成交量强化
# =========================

def volume_relative(df: pd.DataFrame, vol_col: str = "volume", window: int = DEFAULTS["volume_rel_window"]) -> pd.Series:
    rel = df[vol_col] / df[vol_col].rolling(window, min_periods=window).mean()
    return _safe(rel)


def volume_percentile(df: pd.DataFrame, vol_col: str = "volume", window: int = DEFAULTS["volume_pct_window"]) -> pd.Series:
    def _q(x):
        return (x < x.iloc[-1]).mean() if len(x) > 1 else np.nan

    return _safe(
        df[vol_col]
        .rolling(window, min_periods=window)
        .apply(_q, raw=False)
    )


# =========================
# 支撑/阻力 & 距离
# =========================

def support_level(df: pd.DataFrame, low_col: str = "low", window: int = DEFAULTS["support_resist_window"]) -> pd.Series:
    return _safe(df[low_col].rolling(window, min_periods=window).min())


def resistance_level(df: pd.DataFrame, high_col: str = "high", window: int = DEFAULTS["support_resist_window"]) -> pd.Series:
    return _safe(df[high_col].rolling(window, min_periods=window).max())


def distance_to_support(df: pd.DataFrame, price_col: str = "close", support: pd.Series | None = None) -> pd.Series:
    if support is None:
        support = support_level(df)
    return _safe((df[price_col] - support) / df[price_col].clip(lower=1e-9))


def distance_to_resistance(df: pd.DataFrame, price_col: str = "close", resistance: pd.Series | None = None) -> pd.Series:
    if resistance is None:
        resistance = resistance_level(df)
    return _safe((resistance - df[price_col]) / df[price_col].clip(lower=1e-9))


# =========================
# 多周期一致性
# =========================

def trend_agree_multi(df: pd.DataFrame, ma5_cols: dict[str, str], ma20_cols: dict[str, str]) -> pd.Series:
    directions = []
    for tf, ma5_col in ma5_cols.items():
        ma20_col = ma20_cols.get(tf)
        if ma5_col in df and ma20_col in df:
            directions.append(np.sign(df[ma5_col] - df[ma20_col]))
    if not directions:
        return pd.Series(0, index=df.index)
    return _safe(pd.concat(directions, axis=1).sum(axis=1))


# =========================
# 放量关键水位
# =========================

def volume_spike_near_resist(
    df: pd.DataFrame,
    vol_rel_col: str = "volume_relative_20",
    dist_res_col: str = "distance_to_resistance",
    vol_thresh: float = DEFAULTS["volume_spike_thresh"],
    dist_thresh: float = DEFAULTS["distance_thresh"],
) -> pd.Series:
    return ((df[vol_rel_col] > vol_thresh) & (df[dist_res_col] < dist_thresh)).astype(int)


def volume_spike_near_support(
    df: pd.DataFrame,
    vol_rel_col: str = "volume_relative_20",
    dist_sup_col: str = "distance_to_support",
    vol_thresh: float = DEFAULTS["volume_spike_thresh"],
    dist_thresh: float = DEFAULTS["distance_thresh"],
) -> pd.Series:
    return ((df[vol_rel_col] > vol_thresh) & (df[dist_sup_col] < dist_thresh)).astype(int)
