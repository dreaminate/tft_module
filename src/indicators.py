import pandas as pd
import numpy as np

"""indicators_cleaned.py

统一技术指标、信号特征以及关键水位函数。
所有函数均：
- 仅依赖 Pandas / NumPy；
- 遇到 inf / NaN 统一用 `_safe` 清洗；
- 不修改传入 df（无副作用）；
- 返回 `pd.Series`（或 tuple），与 df 索引对齐。

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
    "donchian_window": 20,
    "keltner_period": 20,
    "keltner_mult": 1.5,
    "ppo_short": 12,
    "ppo_long": 26,
    "mfi_period": 14,
    "cmf_period": 20,
    "williams_r_period": 14,
    "tsi_long": 25,
    "tsi_short": 13,
    "ichimoku_conv": 9,
    "ichimoku_base": 26,
    "ichimoku_span_b": 52,
    "supertrend_period": 10,
    "supertrend_mult": 3.0,
}

# =========================
# 工具函数
# =========================
def _safe(series: pd.Series, fill: float | int = 0) -> pd.Series:
    """替换 inf / NaN 为给定值（默认 0）"""
    return series.replace([np.inf, -np.inf], np.nan).fillna(fill)

def _rma(s: pd.Series, period: int) -> pd.Series:
    """Wilder 平滑平均 (RMA)"""
    return s.ewm(alpha=1/period, adjust=False).mean()

def linear_slope(a: np.ndarray) -> float:
    """线性回归斜率"""
    n = len(a)
    x = np.arange(n)
    xm, ym = x.mean(), np.nanmean(a)
    denom = ((x - xm)**2).sum()
    if denom == 0 or np.isnan(ym):
        return 0.0
    return float(((x - xm) * (a - ym)).sum() / denom)

# =========================
# 技术指标系列
# =========================
def calculate_ma(df, window=DEFAULTS["ma_window_short"], column="close"):
    return _safe(df[column].rolling(window=window).mean())

def calculate_rsi(df, period=DEFAULTS["rsi_period"], column="close"):
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return _safe(100 - (100 / (1 + rs)))

def calculate_macd(df, short_window=DEFAULTS["macd_short"], long_window=DEFAULTS["macd_long"],
                   signal_window=DEFAULTS["macd_signal"], column="close"):
    ema_short = df[column].ewm(span=short_window, adjust=False).mean()
    ema_long = df[column].ewm(span=long_window, adjust=False).mean()
    macd_line = _safe(ema_short - ema_long)
    macd_signal = _safe(macd_line.ewm(span=signal_window, adjust=False).mean())
    macd_hist = _safe(macd_line - macd_signal)
    return macd_line, macd_signal, macd_hist

def calculate_kdj(df, period=DEFAULTS["kdj_period"], column="close", high_col="high", low_col="low"):
    low_min = df[low_col].rolling(window=period, min_periods=1).min()
    high_max = df[high_col].rolling(window=period, min_periods=1).max()
    rsv = (df[column] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    return _safe(k), _safe(d), _safe(j)

def calculate_momentum(df, period=DEFAULTS["momentum_period"], column="close"):
    return _safe(df[column] - df[column].shift(period))

def calculate_vol_ma(df, window=DEFAULTS["ma_window_short"], column="volume"):
    return _safe(df[column].rolling(window=window).mean())

def calculate_vol_change(df, column="volume", clip_limit=5.0):
    return _safe(df[column].pct_change().clip(-clip_limit, clip_limit))

def calculate_obv(df, column_price="close", column_volume="volume"):
    direction = np.sign(df[column_price].diff()).fillna(0)
    return _safe((direction * df[column_volume]).cumsum())

def calculate_bollinger(df, period=DEFAULTS["boll_period"], column="close"):
    ma = _safe(df[column].rolling(window=period).mean())
    std = df[column].rolling(window=period).std()
    return ma, _safe(ma + 2 * std), _safe(ma - 2 * std)

def calculate_atr(df, period=DEFAULTS["atr_period"], high_col="high", low_col="low", close_col="close"):
    tr = pd.concat([
        df[high_col] - df[low_col],
        (df[high_col] - df[close_col].shift()).abs(),
        (df[low_col] - df[close_col].shift()).abs(),
    ], axis=1).max(axis=1)
    return _safe(tr.rolling(window=period, min_periods=1).mean())

def calculate_cci(df, period=DEFAULTS["cci_period"], high_col="high", low_col="low", close_col="close"):
    tp = (df[high_col] + df[low_col] + df[close_col]) / 3
    sma = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return _safe((tp - sma) / (0.015 * mad + 1e-10))

def calculate_adx(df, period=DEFAULTS["adx_period"], high_col="high", low_col="low", close_col="close"):
    idx = df.index
    up_move = df[high_col].diff()
    down_move = df[low_col].shift() - df[low_col]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        df[high_col] - df[low_col],
        (df[high_col] - df[close_col].shift()).abs(),
        (df[low_col] - df[close_col].shift()).abs(),
    ], axis=1).max(axis=1)

    atr = _rma(tr, period)
    plus_di = 100.0 * _rma(pd.Series(plus_dm, index=idx), period) / (atr + 1e-10)
    minus_di = 100.0 * _rma(pd.Series(minus_dm, index=idx), period) / (atr + 1e-10)

    dx = 100.0 * (plus_di.sub(minus_di).abs() / (plus_di.add(minus_di) + 1e-10))
    adx = _rma(dx, period)

    return (
        _safe(adx).clip(0, 100).astype("float32"),
        _safe(plus_di).clip(0, 100).astype("float32"),
        _safe(minus_di).clip(0, 100).astype("float32")
    )

def calculate_vwap(df, period=None, column_price="close", column_volume="volume", high_col="high", low_col="low"):
    tp = (df[high_col] + df[low_col] + df[column_price]) / 3
    if period is None:
        return _safe((tp * df[column_volume]).cumsum() / df[column_volume].cumsum())
    pv = (tp * df[column_volume]).rolling(window=period, min_periods=1).sum()
    v = df[column_volume].rolling(window=period, min_periods=1).sum()
    return _safe(pv / v)


# =========================
# 高阶指标扩展
# =========================
def donchian_channel(df: pd.DataFrame, window: int = DEFAULTS["donchian_window"], high_col: str = "high", low_col: str = "low") -> tuple[pd.Series, pd.Series]:
    up = _safe(df[high_col].rolling(window, min_periods=1).max())
    low = _safe(df[low_col].rolling(window, min_periods=1).min())
    return up, low


def keltner_channel(
    df: pd.DataFrame,
    period: int = DEFAULTS["keltner_period"],
    multiplier: float = DEFAULTS["keltner_mult"],
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema = _safe(df[price_col].ewm(span=period, adjust=False).mean())
    atr = calculate_atr(df, period=period, high_col=high_col, low_col=low_col, close_col=price_col)
    upper = _safe(ema + multiplier * atr)
    lower = _safe(ema - multiplier * atr)
    return ema, upper, lower


def calculate_ppo(df: pd.DataFrame, short: int = DEFAULTS["ppo_short"], long: int = DEFAULTS["ppo_long"], column: str = "close") -> pd.Series:
    ema_s = df[column].ewm(span=short, adjust=False).mean()
    ema_l = df[column].ewm(span=long, adjust=False).mean()
    ppo = (ema_s - ema_l) / (ema_l.replace(0, np.nan) + 1e-10)
    return _safe(ppo)


def stoch_rsi(df: pd.DataFrame, period: int = DEFAULTS["rsi_period"], column: str = "close") -> pd.Series:
    rsi = calculate_rsi(df, period=period, column=column)
    lowest = rsi.rolling(period, min_periods=1).min()
    highest = rsi.rolling(period, min_periods=1).max()
    return _safe((rsi - lowest) / (highest - lowest + 1e-10))


def williams_r(df: pd.DataFrame, period: int = DEFAULTS["williams_r_period"], high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.Series:
    hh = df[high_col].rolling(period, min_periods=1).max()
    ll = df[low_col].rolling(period, min_periods=1).min()
    wr = -100.0 * (hh - df[close_col]) / (hh - ll + 1e-10)
    return _safe(wr)


def tsi(df: pd.DataFrame, long: int = DEFAULTS["tsi_long"], short: int = DEFAULTS["tsi_short"], column: str = "close") -> pd.Series:
    m = df[column].diff()
    m_ema1 = m.ewm(span=long, adjust=False).mean()
    m_ema2 = m_ema1.ewm(span=short, adjust=False).mean()
    am = m.abs()
    am_ema1 = am.ewm(span=long, adjust=False).mean()
    am_ema2 = am_ema1.ewm(span=short, adjust=False).mean()
    return _safe(m_ema2 / (am_ema2 + 1e-10))


def ichimoku(df: pd.DataFrame, conv: int = DEFAULTS["ichimoku_conv"], base: int = DEFAULTS["ichimoku_base"], span_b: int = DEFAULTS["ichimoku_span_b"], high_col: str = "high", low_col: str = "low") -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    conv_line = _safe((df[high_col].rolling(conv, min_periods=1).max() + df[low_col].rolling(conv, min_periods=1).min()) / 2.0)
    base_line = _safe((df[high_col].rolling(base, min_periods=1).max() + df[low_col].rolling(base, min_periods=1).min()) / 2.0)
    span_a = _safe(((conv_line + base_line) / 2.0).shift(base))
    span_b_line = _safe(((df[high_col].rolling(span_b, min_periods=1).max() + df[low_col].rolling(span_b, min_periods=1).min()) / 2.0).shift(base))
    return conv_line, base_line, span_a, span_b_line


def psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2, high_col: str = "high", low_col: str = "low") -> tuple[pd.Series, pd.Series]:
    high = df[high_col].to_numpy()
    low = df[low_col].to_numpy()
    n = len(df)
    psar = np.zeros(n, dtype=float)
    bull = True
    af = step
    ep = low[0]
    hp = high[0]
    lp = low[0]
    for i in range(1, n):
        prev = psar[i-1]
        if bull:
            psar[i] = prev + af * (hp - prev)
            psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
            if high[i] > hp:
                hp = high[i]
                af = min(max_step, af + step)
            if low[i] < psar[i]:
                bull = False
                psar[i] = hp
                lp = low[i]
                af = step
        else:
            psar[i] = prev + af * (lp - prev)
            psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
            if low[i] < lp:
                lp = low[i]
                af = min(max_step, af + step)
            if high[i] > psar[i]:
                bull = True
                psar[i] = lp
                hp = high[i]
                af = step
    psar_series = _safe(pd.Series(psar, index=df.index))
    flip = (psar_series.shift(1) > df[high_col]) & (psar_series <= df[low_col]) | (psar_series.shift(1) < df[low_col]) & (psar_series >= df[high_col])
    return psar_series, _safe(flip.astype(int))


def supertrend(df: pd.DataFrame, period: int = DEFAULTS["supertrend_period"], multiplier: float = DEFAULTS["supertrend_mult"], high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.Series:
    atr = calculate_atr(df, period=period, high_col=high_col, low_col=low_col, close_col=close_col)
    hl2 = (df[high_col] + df[low_col]) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    st = np.zeros(len(df), dtype=float)
    up_trend = True
    for i in range(1, len(df)):
        if df[close_col].iloc[i] > upper.iloc[i-1]:
            up_trend = True
        elif df[close_col].iloc[i] < lower.iloc[i-1]:
            up_trend = False
        st[i] = lower.iloc[i] if up_trend else upper.iloc[i]
    return _safe(pd.Series(st, index=df.index))


def heikin_ashi_trend(df: pd.DataFrame) -> pd.Series:
    close = df[["open", "high", "low", "close"]].mean(axis=1)
    open_ha = (df["open"].shift(1) + df["close"].shift(1)) / 2.0
    open_ha = open_ha.fillna(df["open"]) 
    return _safe(close - open_ha)


def money_flow_index(df: pd.DataFrame, period: int = DEFAULTS["mfi_period"], high_col: str = "high", low_col: str = "low", close_col: str = "close", volume_col: str = "volume") -> pd.Series:
    tp = (df[high_col] + df[low_col] + df[close_col]) / 3.0
    mf = tp * df[volume_col]
    pos_mf = np.where(tp > tp.shift(1), mf, 0.0)
    neg_mf = np.where(tp < tp.shift(1), mf, 0.0)
    pos_sum = pd.Series(pos_mf, index=df.index).rolling(period, min_periods=1).sum()
    neg_sum = pd.Series(neg_mf, index=df.index).rolling(period, min_periods=1).sum()
    mr = pos_sum / (neg_sum + 1e-10)
    mfi = 100.0 - (100.0 / (1.0 + mr))
    return _safe(mfi)


def chaikin_money_flow(df: pd.DataFrame, period: int = DEFAULTS["cmf_period"], high_col: str = "high", low_col: str = "low", close_col: str = "close", volume_col: str = "volume") -> pd.Series:
    mfm = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / ((df[high_col] - df[low_col]).replace(0, np.nan) + 1e-10)
    mfv = mfm * df[volume_col]
    return _safe(mfv.rolling(period, min_periods=1).sum() / (df[volume_col].rolling(period, min_periods=1).sum() + 1e-10))


def price_volume_corr(df: pd.DataFrame, window: int = 20, close_col: str = "close", volume_col: str = "volume") -> pd.Series:
    r = np.log(df[close_col].replace(0, np.nan)).diff()
    dv = df[volume_col].pct_change().replace([np.inf, -np.inf], np.nan)
    return _safe(r.rolling(window, min_periods=2).corr(dv))


def pivot_point(df: pd.DataFrame, high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.Series:
    return _safe((df[high_col] + df[low_col] + df[close_col]) / 3.0)


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

def linear_slope(a: np.ndarray) -> float:
    n = len(a); x = np.arange(n)
    xm, ym = x.mean(), np.nanmean(a)
    denom = ((x - xm)**2).sum()
    if denom == 0 or np.isnan(ym): return 0.0
    return float(((x - xm) * (a - ym)).sum() / denom)

def obv_diff(df, col="obv"):
    return _safe(df[col].diff())

def obv_pct_change(df, col="obv", clip=5.0):
    return _safe(df[col].pct_change().clip(-clip, clip))

def obv_slope(df, w=14, col="obv"):
    return _safe(df[col].rolling(w, min_periods=w).apply(linear_slope, raw=True))

def atr_ratio_price(df, atr_col="atr", price_col="close"):
    return _safe(df[atr_col] / (df[price_col].abs() + 1e-9))

def atr_ratio_range(df, atr_col="atr", high_col="high", low_col="low"):
    return _safe(df[atr_col] / ((df[high_col]-df[low_col]).abs() + 1e-9))

def atr_change(df, atr_col="atr", clip=2.0):
    return _safe(df[atr_col].pct_change().clip(-clip, clip))

def range_to_atr(df, high_col="high", low_col="low", atr_col="atr"):
    return _safe((df[high_col]-df[low_col]) / (df[atr_col] + 1e-9))

def boll_pctb(df, close_col="close", up_col="boll_upper", low_col="boll_lower"):
    width = (df[up_col] - df[low_col]).abs()
    return _safe((df[close_col] - df[low_col]) / (width + 1e-9))

def boll_bandwidth(df, ma_col="ma20", up_col="boll_upper", low_col="boll_lower"):
    return _safe((df[up_col] - df[low_col]) / (df[ma_col].abs() + 1e-9))