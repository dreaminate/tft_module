# ✅ indicators_cleaned.py — 清理字段后缀、用于长格式字段统一的指标函数
import pandas as pd 
import numpy as np

def _safe(series: pd.Series, fill=0):
    return series.replace([np.inf, -np.inf], np.nan).fillna(fill)

def calculate_ma(df, window=5, column='close'):
    return _safe(df[column].rolling(window=window).mean())

def calculate_rsi(df, period=14, column='close'):
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs  = avg_gain / (avg_loss.replace(0, 1e-10))
    rsi = 100 - (100 / (1 + rs))
    return _safe(rsi)

def calculate_macd(df, short_window=12, long_window=26, signal_window=9, column='close'):
    ema_short = df[column].ewm(span=short_window, adjust=False).mean()
    ema_long  = df[column].ewm(span=long_window,  adjust=False).mean()
    macd_line   = _safe(ema_short - ema_long)
    macd_signal = _safe(macd_line.ewm(span=signal_window, adjust=False).mean())
    macd_hist   = _safe(macd_line - macd_signal)
    return macd_line, macd_signal, macd_hist

def calculate_kdj(df, period=9, column='close', high_col='high', low_col='low'):
    low_min  = df[low_col].rolling(window=period, min_periods=1).min()
    high_max = df[high_col].rolling(window=period, min_periods=1).max()
    rsv = (df[column] - low_min) / (high_max - low_min).replace(0, pd.NA) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    return _safe(k), _safe(d), _safe(j)

def calculate_momentum(df, period=5, column='close'):
    return _safe(df[column] - df[column].shift(period))

def calculate_vol_ma(df, window=5, column='volume'):
    return _safe(df[column].rolling(window=window).mean())

def calculate_vol_change(df, column='volume', clip_limit=5.0):
    change = df[column].pct_change()
    change = change.clip(lower=-clip_limit, upper=clip_limit)
    return _safe(change)

def calculate_obv(df, column_price='close', column_volume='volume'):
    obv = [0]
    for i in range(1, len(df)):
        if df[column_price].iloc[i] > df[column_price].iloc[i - 1]:
            obv.append(obv[-1] + df[column_volume].iloc[i])
        elif df[column_price].iloc[i] < df[column_price].iloc[i - 1]:
            obv.append(obv[-1] - df[column_volume].iloc[i])
        else:
            obv.append(obv[-1])
    return _safe(pd.Series(obv, index=df.index))

def calculate_bollinger(df, period=20, column='close'):
    ma  = _safe(df[column].rolling(window=period).mean())
    std = df[column].rolling(window=period).std()
    upper = _safe(ma + 2 * std)
    lower = _safe(ma - 2 * std)
    return ma, upper, lower

def calculate_atr(df, period=14, high_col='high', low_col='low', close_col='close'):
    tr = pd.concat([
        df[high_col] - df[low_col],
        (df[high_col] - df[close_col].shift()).abs(),
        (df[low_col]  - df[close_col].shift()).abs()], axis=1).max(axis=1)
    atr = _safe(tr.rolling(window=period, min_periods=1).mean())
    return atr

def calculate_cci(df, period=20, high_col='high', low_col='low', close_col='close'):
    tp  = (df[high_col] + df[low_col] + df[close_col]) / 3
    sma = tp.rolling(window=period, min_periods=1).mean()
    mad = tp.rolling(window=period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-10)
    return _safe(cci)

def calculate_adx(df, period=14, high_col='high', low_col='low', close_col='close'):
    df = df.copy()
    df['up_move']   = df[high_col].diff()
    df['down_move'] = df[low_col].shift() - df[low_col]
    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move']  > 0), df['up_move'],  0.0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    tr = pd.concat([
        df[high_col] - df[low_col],
        (df[high_col] - df[close_col].shift()).abs(),
        (df[low_col]  - df[close_col].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di  = 100 * (df['+dm'].ewm(alpha=1/period).mean() / (atr + 1e-10))
    minus_di = 100 * (df['-dm'].ewm(alpha=1/period).mean() / (atr + 1e-10))
    dx  = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(alpha=1/period).mean()
    return _safe(adx), _safe(plus_di), _safe(minus_di)

def calculate_vwap(df, period=None, column_price='close', column_volume='volume', high_col='high', low_col='low'):
    tp = (df[high_col] + df[low_col] + df[column_price]) / 3
    if period is None:
        pv = (tp * df[column_volume]).cumsum()
        v  = df[column_volume].cumsum()
        df['vwap'] = pv / v
    else:
        pv = (tp * df[column_volume]).rolling(window=period, min_periods=1).sum()
        v  = df[column_volume].rolling(window=period, min_periods=1).sum()
        df[f'vwap_{period}'] = pv / v
    for col in [c for c in df.columns if "vwap" in c]:
        df[col] = _safe(df[col])
    return df
