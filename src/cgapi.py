#      python src/cgapi.py
import os
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Union, List
import numpy as np
# 所有函数名汇总
'''
    "_infer_default_window","_ensure_numeric","_roll_zscore","_roll_robust","_roll_minmax","_to_ms","fetch_funding_rate",
    "accum_as_of","long_short_account_ratio","fetch_oi","taker_buy_sell_volume","premium_index","fetch_index_AHR999",
    "fetch_index_fear_greed","fetch_margin_long_short","fetch_borrow_interest_rate","fetch_puell_multiple","fetch_stock_flow",
    "fetch_pi_cycle_indicator","fetch_golden_ratio_multiplier","fetch_profitable_days","fetch_rainbow_chart","fetch_stableCoin_marketCap_history",
    "fetch_bubble_index","fetch_altcoin_season","fetch_bitcoin_sth_sopr","fetch_bitcoin_lth_sopr","fetch_bitcoin_sth_realized_price",
    "fetch_bitcoin_lth_realized_price","fetch_bitcoin_short_term_holder_supply","fetch_bitcoin_long_term_holder_supply","fetch_bitcoin_rhodl_ratio",
    "fetch_bitcoin_new_addresses","fetch_bitcoin_active_addresses","fetch_bitcoin_reserve_risk","fetch_bitcoin_net_unrealized_profit_loss",
    "fetch_bitcoin_correlation","fetch_bitcoin_macro_oscillator","fetch_bitcoin_vs_global_m2_growth","fetch_bitcoin_vs_us_m2_growth",
    "fetch_bitcoin_dominance",
'''
API_KEY = "5bef675000e144fcb6d9fe9960129d79"
BASE = "https://open-api-v4.coinglass.com"
def _infer_default_window(interval: str) -> int:
    """
    根据你的训练设定给出默认滚动窗口：
    1h -> 96, 4h -> 48, 1d -> 30，其他周期回退到 96。
    """
    key = (interval or "").lower()
    if key == "1h":
        return 96
    if key == "4h":
        return 48
    if key == "1d":
        return 30
    return 96

def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _roll_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=min_periods).mean().shift(1)
    std  = series.rolling(window=window, min_periods=min_periods).std(ddof=0).shift(1)
    return (series - mean) / (std.replace(0.0, np.nan))

def _roll_robust(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    med = series.rolling(window=window, min_periods=min_periods).median().shift(1)
    q25 = series.rolling(window=window, min_periods=min_periods).quantile(0.25).shift(1)
    q75 = series.rolling(window=window, min_periods=min_periods).quantile(0.75).shift(1)
    iqr = (q75 - q25).replace(0.0, np.nan)
    return (series - med) / iqr

def _roll_minmax(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    rmin = series.rolling(window=window, min_periods=min_periods).min().shift(1)
    rmax = series.rolling(window=window, min_periods=min_periods).max().shift(1)
    return (series - rmin) / ((rmax - rmin).replace(0.0, np.nan))
def _to_ms(t: Union[str, int, float, datetime, None]) -> Optional[int]:
    """把时间统一成毫秒整数（UTC）。"""
    if t is None:
        return None
    if isinstance(t, (int, float)):
        v = int(t)
        return v if v >= 10**12 else v * 1000
    if isinstance(t, datetime):
        dt = t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if isinstance(t, str):
        s = t.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            s += "T00:00:00+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    raise TypeError(f"不支持的时间类型: {type(t)}")
def fetch_funding_rate(
    symbol: str,
    interval: str = "4h",
    limit: int = 1000,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    ranged:  Optional[str] = None,
    base_dir: str = "data/cglass/futures/funding-rate",
    
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    直接请求 CoinGlass 的“加权资金费率”两条序列并保存：
      - data/cglass/funding-rate/funding-rate-oi/Agg_{symbol}_{interval}.csv
      - data/cglass/funding-rate/funding-rate-volume/Agg_{symbol}_{interval}.csv
      - data/cglass/funding-rate/funding-rate-accumulated/Agg_{symbol}_{interval}.csv

    返回: (df_oi, df_vol, df_accum, paths)
    """

    
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    def _fetch_weighted(path: str) -> pd.DataFrame:
        url = BASE + path
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,  # 默认值为 1000，最大值为 4500。
            "start_time": start_time,
            "end_time": end_time
        }
        if start_time is not None:
            params["start_time"] = int(start_time)  # ms
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"API error: {payload}")
        df = pd.DataFrame(payload["data"])
        # 规范化字段
        if "time" in df.columns:
            df.rename(columns={"time": "timestamp"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for c in ("open", "high", "low", "close"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["symbol"] = symbol
        df["period"] = interval
        df["symbol"] = symbol+"_USDT"
        return df.sort_values("timestamp")
    def _fetch_accumulated(path: str) -> pd.DataFrame:
        url = BASE + path
        params = {"range": ranged}  # 例如 "1d"/"7d"/"30d"/"365d"
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"API error: {payload}")

        rows = []
        for item in payload.get("data", []):
            sym = item.get("symbol")
            # USDT/USD 保证金
            for rec in (item.get("stablecoin_margin_list") or []):
                rows.append({
                    "symbol": sym,
                    "exchange": rec.get("exchange"),
                    "margin_type": "stablecoin",
                    "funding_rate_accum": pd.to_numeric(rec.get("funding_rate"), errors="coerce"),
                })
            # 币本位保证金
            for rec in (item.get("token_margin_list") or []):
                rows.append({
                    "symbol": sym,
                    "exchange": rec.get("exchange"),
                    "margin_type": "token",
                    "funding_rate_accum": pd.to_numeric(rec.get("funding_rate"), errors="coerce"),
                })

        if not rows:
            # 返回空表但列名齐全，避免后续 KeyError
            return pd.DataFrame(columns=[
                "symbol","exchange","margin_type","funding_rate_accum","range","fetched_at"
            ])

        df = pd.DataFrame(rows)
        df["range"] = ranged
        df["fetched_at"] = pd.Timestamp.utcnow()
        
        # 统一顺序输出（无 time 列）
        return df[["symbol","exchange","margin_type","funding_rate_accum","range","fetched_at"]] \
                .sort_values(["symbol","exchange","margin_type"])
    
    # 1) OI 加权
    df_oi = _fetch_weighted("/api/futures/funding-rate/oi-weight-history")
    out_dir_oi = os.path.join(base_dir, "funding-rate-oi")
    os.makedirs(out_dir_oi, exist_ok=True)
    oi_path = os.path.join(out_dir_oi, f"Agg_{symbol}_{interval}.csv")
    df_oi.to_csv(oi_path, index=False, encoding="utf-8-sig")

    # 2) 成交量加权
    df_vol = _fetch_weighted("/api/futures/funding-rate/vol-weight-history")
    out_dir_vol = os.path.join(base_dir, "funding-rate-volume")
    os.makedirs(out_dir_vol, exist_ok=True)
    vol_path = os.path.join(out_dir_vol, f"Agg_{symbol}_{interval}.csv")
    df_vol.to_csv(vol_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved: {oi_path}  |  {vol_path}")

    # 3) 累计资金费率
    if ranged  not in ("1h", "4h") and ranged is not None:
        df_accum = _fetch_accumulated("/api/futures/funding-rate/accumulated-exchange-list")
        out_dir_accum = os.path.join(base_dir, "funding-rate-accumulated-exchange-list")
        os.makedirs(out_dir_accum, exist_ok=True)
        accum_path = os.path.join(out_dir_accum, f"Agg_ALL_{ranged}.csv")
        df_accum.to_csv(accum_path, index=False, encoding="utf-8-sig")
        return df_oi, df_vol, df_accum, {"oi": oi_path, "volume": vol_path, "accumulated": accum_path}
    else:
        print(f"⚠️ Skipped accumulated funding rate for ALL at {ranged} interval.")
        return df_oi, df_vol,  {"oi": oi_path, "volume": vol_path}
def accum_as_of(symbol="BTC", interval="4h", window="7d",
                kind="oi",  # "oi" 或 "volume"
                asof=None,
                base_dir="data/cglass"):
    folder = "funding-rate-oi" if kind=="oi" else "funding-rate-volume"
    fp = os.path.join(base_dir, folder, f"Agg_{symbol}_{interval}.csv")
    df = pd.read_csv(fp, parse_dates=["time"])
    if asof is None:
        asof = df["time"].max()
    else:
        asof = pd.to_datetime(asof, utc=True)

    start = asof - pd.to_timedelta(window)
    cut = df[(df["time"] > start) & (df["time"] <= asof)].copy()
    if cut.empty:
        return np.nan

    r = pd.to_numeric(cut.get("close"), errors="coerce").fillna(0.0)
    # 复利累计：(1+r)连乘 - 1；r 很小用加和近似也行
    acc = np.exp(np.log1p(r).sum()) - 1
    return float(acc)
# a=fetch_funding_rate("BTC",interval="1h",ranged="365d",start_time=1751385600000,end_time=1754982000000)
# print(a[0].head())
# print(a[1].head())
def long_short_account_ratio(
    symbol: str,            # 例如: "BTCUSDT"
    exchange: str = "Binance",          # 例如: "Binance" / "OKX"
    interval: str = "4h",
    limit: int = 1000,
    start_time: Optional[int] = None,   # ms
    end_time: Optional[int] = None,     # ms
    base_dir: str = "data/cglass/futures/funding-rate",  # 会自动回到 data/cglass 根目录
    is_now: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:

    
    
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}

    def _fetch(path: str) -> pd.DataFrame:
        INT_MS = {"1h": 3600_000, "4h": 14_400_000, "1d": 86_400_000}
        def _align_floor(ts_ms: int, step: int) -> int:
            return (int(ts_ms) // step) * step
        url = BASE + path
        step = INT_MS.get(interval, 14_400_000)

        ex = (exchange or "").upper()          # 有些端点更偏好大写
        sym = symbol.upper()                    # 这里确实需要交易对，如 BTCUSDT

        base_params = {"symbol": sym, "exchange": ex, "interval": interval, "limit": limit}

        # 组 3 套参数逐一尝试
        attempts = []

        if start_time is not None and end_time is not None:
            st = _align_floor(int(start_time), step)
            et = _align_floor(int(end_time), step)
            if et <= st: et = st + step
            if et - st < step: et = st + step
            p_both = {**base_params, "start_time": st, "end_time": et}
            attempts.append(p_both)

        if start_time is not None:
            st = _align_floor(int(start_time), step)
            p_start = {**base_params, "start_time": st}
            attempts.append(p_start)

        # 最后兜底：只用 limit（最近 N 根）
        attempts.append(base_params)

        last_payload = None
        for params in attempts:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            try:
                r.raise_for_status()
                payload = r.json()
                if str(payload.get("code")) == "0":
                    df = pd.DataFrame(payload.get("data", []))
                    # 统一列名
                    df.rename(columns={"time": "timestamp"}, inplace=True)
                    return df
                last_payload = payload
            except requests.HTTPError:
                last_payload = {"code": str(r.status_code), "msg": r.text[:200]}
                continue

        raise RuntimeError(f"API error after fallbacks: {last_payload}")

    # 拉取 3 个序列
    df_global = _fetch("/api/futures/global-long-short-account-ratio/history")
    df_top_acc = _fetch("/api/futures/top-long-short-account-ratio/history")
    df_top_pos = _fetch("/api/futures/top-long-short-position-ratio/history")

    # 计算根目录（把你传的 base_dir 回退到 data/cglass）
    root = os.path.dirname(os.path.dirname(base_dir))  # e.g. data/cglass/futures/funding-rate -> data/cglass
    out_dirs = {
        "global": os.path.join(root, "futures/long-short/global-long-short-account-ratio"),
        "top_account": os.path.join(root, "futures/long-short/top-long-short-account-ratio"),
        "top_position": os.path.join(root, "futures/long-short/top-long-short-position-ratio"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    paths: Dict[str, str] = {}
    if df_global is not None and not df_global.empty:
        paths["global"] = os.path.join(out_dirs["global"], f"{exchange}_{symbol}_{interval}.csv")
        df_global.to_csv(paths["global"], index=False, encoding="utf-8-sig")
    if df_top_acc is not None and not df_top_acc.empty:
        paths["top_account"] = os.path.join(out_dirs["top_account"], f"{exchange}_{symbol}_{interval}.csv")
        df_top_acc.to_csv(paths["top_account"], index=False, encoding="utf-8-sig")
    if df_top_pos is not None and not df_top_pos.empty:
        paths["top_position"] = os.path.join(out_dirs["top_position"], f"{exchange}_{symbol}_{interval}.csv")
        df_top_pos.to_csv(paths["top_position"], index=False, encoding="utf-8-sig")

    return df_global, df_top_acc, df_top_pos, paths
# a,b,c,d=long_short_account_ratio(symbol="BTCUSDT",exchange="Binance",interval="4h",limit=1000,start_time = 1641513600000)
# print(a.head())
def fetch_oi(
    symbol: str,
    interval: str = "4h",
    exchange_list: Union[List[str], str, None] = None,
    limit: int = 1000,
    start_time: Optional[Union[str, int, float, datetime]] = None,
    end_time: Optional[Union[str, int, float, datetime]] = None,
    ranged: Optional[str] = None,
    base_dir: str = "data/cglass/futures/open-interest",
) -> pd.DataFrame:
    # --- 规范 exchange_list 参数与文件名片段 ---
    if exchange_list is None:
        
        exchange_list ="Binance,OKX,Bybit"

    

    params = {
        "exchange_list": exchange_list,   # 该端点为必填
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time":  start_time,
        "end_time": end_time
    }

    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/futures/open-interest/aggregated-stablecoin-history"
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    
    payload = r.json()
    
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"API error: {payload}")
    
    # --- 解析为规范 DataFrame ---
    df = pd.DataFrame(payload["data"])
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c+"_oi"] = pd.to_numeric(df[c], errors="coerce")

    df["symbol"] = symbol+"_USDT"
    df["period"] = interval
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df = df.sort_values(["symbol", "period", "timestamp"]).reset_index(drop=True)
    
    # === 子函数：添加 OI 特征（变化率、ATR-like 波动率、rolling 标准化） ===
    def _add_oi_features(_df: pd.DataFrame, _interval: str) -> pd.DataFrame:
        eps = 1e-12
        win_map = {"1h": 96, "2h": 72, "4h": 48, "6h": 36, "8h": 30, "12h": 24, "1d": 30, "3d": 20, "1w": 26}
        W = int(win_map.get(_interval, 48))

        # 兜底 open/close
        oi_open = _df["open"] if "open" in _df.columns else _df["close"]
        oi_close = _df["close"] if "close" in _df.columns else oi_open

        # 1) 开收均值
        _df["oi_mean_oc"] = (oi_open.fillna(oi_close) + oi_close) / 2.0

        g = _df.groupby(["symbol", "period"], sort=False)

        # 2) 变化率（log-return）
        prev_close = _df.groupby(["symbol", "period"])["close"].shift(1)
        _df["oi_logret"] = np.log(oi_close.clip(lower=eps) / prev_close.clip(lower=eps))

        # 3) 波动率（优先 ATR-like；若缺 high/low 则回退为 logret 的滚动 std）
        if ("high" in _df.columns) and ("low" in _df.columns):
            tr = pd.concat([
                (_df["high"] - _df["low"]).abs(),
                (_df["high"] - prev_close).abs(),
                (_df["low"]  - prev_close).abs(),
            ], axis=1).max(axis=1)
            _df["oi_vol"] = tr.groupby([_df["symbol"], _df["period"]]).transform(
                lambda s: s.rolling(W, min_periods=max(5, W // 3)).mean()
            )
        else:
            _df["oi_vol"] = g["oi_logret"].transform(
                lambda s: s.rolling(W, min_periods=max(5, W // 3)).std(ddof=0)
            )

        # 4) oi_mean_oc 的滚动 z-score（无泄露：rolling + shift(1)）
        def _rolling_z_series(s: pd.Series, W: int):
            roll = s.rolling(W, min_periods=max(5, W // 3))
            mean_prev = roll.mean().shift(1)
            std_prev = roll.std(ddof=0).shift(1)
            std_prev = std_prev.where(std_prev > 0, np.nan)
            return (s - mean_prev) / (std_prev + eps)

        _df["oi_mean_oc_z"] = g["oi_mean_oc"].transform(lambda s: _rolling_z_series(s, W))

        # 起始段填NaN
        _df[["oi_logret", "oi_vol", "oi_mean_oc_z"]] = _df[["oi_logret", "oi_vol", "oi_mean_oc_z"]].fillna(np.nan)
        
        return _df

    df = _add_oi_features(df, interval)


    df.drop(columns=["high","low","close","open"], inplace=True, errors="ignore")

    # --- 保存 CSV ---
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f"{symbol}_USDT_{interval}.csv")
    df.to_csv(out_path, index=False)

    return df
# a=fetch_oi(symbol="BTC", interval="1d", limit=1000, start_time=1641513600000)
# print(a.head())
def taker_buy_sell_volume(
       
        symbol: str,
        interval: str,
        limit: int,
        start_time: int,
        end_time: int,
        *,
        exchange: str = "Binance",
        standardize: bool = True,
        method: str = "zscore_rolling",     # "zscore_rolling" | "robust_rolling" | "minmax_rolling"
        window: int | None = None,          # None 时按 interval 自动推断
        min_periods: int | None = None,     # None -> max(8, window//4)
        add_imbalance: bool = True,         # 是否添加买卖力量差 (buy-sell)/(buy+sell)
        keep_raw: bool = True,               # 是否保留原始列
        base_dir: str = "data/cglass/spot/taker-buy-sell-volume"
    ) -> pd.DataFrame:

    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/spot/taker-buy-sell-volume/history"
    params = {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    js = resp.json()

    df = pd.DataFrame(js.get("data", []))
    

    # 基础列 & 排序
    df = df.sort_values("time").reset_index(drop=True)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    _ensure_numeric(df, ["timestamp", "taker_buy_volume_usd", "taker_sell_volume_usd"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["exchange"] = exchange
    df["symbol"] = symbol
    df["period"] = interval

    # 买卖力量差
    if add_imbalance:
        eps = 1e-12
        den = (df["taker_buy_volume_usd"].astype(float) + df["taker_sell_volume_usd"].astype(float)) + eps
        df["taker_imbalance"] = (df["taker_buy_volume_usd"] - df["taker_sell_volume_usd"]) / den

    # 标准化（无未来泄露）
    if standardize:
        win = window if window is not None else _infer_default_window(interval)
        mp = max(8, win // 4) if min_periods is None else min_periods

        target_cols = ["taker_buy_volume_usd", "taker_sell_volume_usd"]
        if add_imbalance:
            target_cols.append("taker_imbalance")

        method_code = {"zscore_rolling": "z", "robust_rolling": "r", "minmax_rolling": "m"}
        if method not in method_code:
            raise ValueError(f"Unknown method: {method}")
        code = method_code[method]

        created_std_cols = []

        for col in target_cols:
            if col not in df.columns:
                continue
            s = df[col].astype(float)

            if method == "zscore_rolling":
                ser = _roll_zscore(s, win, mp)
            elif method == "robust_rolling":
                ser = _roll_robust(s, win, mp)
            elif method == "minmax_rolling":
                ser = _roll_minmax(s, win, mp)

            new_col = f"{col}_{code}"   # 仅 z/r/m
            df[new_col] = ser
            created_std_cols.append(new_col)

        if not keep_raw:
            base_cols = ["timestamp", "datetime", "exchange", "symbol", "period"]
            df = df[base_cols + created_std_cols]
    # --- 保存 CSV ---
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f"{symbol}_{interval}.csv")
    df.to_csv(out_path, index=False)

    return df
# a = taker_buy_sell_volume("BTCUSDT", "1h", 1000, 1751385600000, 1754982000000)
# print(a[["timestamp", "datetime"]].head())
def premium_index(
        interval: str = "1d",
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None
) -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/coinbase-premium-index"
    params = {
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    js = resp.json()

    df = pd.DataFrame(js.get("data", []))
    df = df.sort_values("time").reset_index(drop=True)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if df["timestamp"].max() < 1_000_000_000_000:  # < 1e12 说明是“秒”
        df["timestamp"] = (df["timestamp"] * 1000).astype("int64")  # 统一成“毫秒”
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["period"] = interval
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/premium-index.csv", index=False)
    return df
# a = premium_index("1h", 1000, 1751385600000, 1754982000000)
# print(a.head())
def fetch_index_AHR999() -> pd.DataFrame:
    """
    拉取 Coinglass AHR999 指标（日频），返回包含以下列的 DataFrame：
      - date_string: 原始日期字符串 (YYYY/MM/DD)
      - average_price: 当日平均价 (float)
      - ahr999_value: AHR999 指数值 (float)
      - current_value: 当日当前值 (float)
      - datetime: UTC 时间（当天 00:00:00），pandas datetime64[ns, UTC]
      - timestamp: 毫秒时间戳 (int)
    """
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = f"{BASE}/api/index/ahr999"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    # 基本校验
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"API error: code={payload.get('code')}, msg={payload.get('msg')}")

    data = payload.get("data")
    if not data:
        # 返回一个空表但保留列结构
        return pd.DataFrame(columns=[
            "date_string", "average_price", "ahr999_value", "current_value",
            "datetime", "timestamp"
        ])

    df = pd.DataFrame(data)

    # 类型安全转换
    for col in ["average_price", "ahr999_value", "current_value"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # 解析 YYYY/MM/DD -> UTC datetime（当天 00:00）
    # 注意：这里不是毫秒时间戳，所以不能直接 to_numeric。
    dt = pd.to_datetime(df["date_string"], format="%Y/%m/%d", utc=True)
    df["datetime"] = dt

    # 生成毫秒时间戳
    df["timestamp"] = (df["datetime"].view("int64") // 10**6).astype("int64")

    # 排序、去重
    df = (
        df.sort_values("datetime")
          .drop_duplicates(subset=["datetime"], keep="last")
          .reset_index(drop=True)
    )
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/ahr999.csv", index=False)

    # 返回常用列顺序
    return df
# a = fetch_index_AHR999()
# print(a.head())
def fetch_index_fear_greed() -> pd.DataFrame:
    """
    拉取 Coinglass 恐惧与贪婪指数（日频）并返回 DataFrame，列包含：
      - timestamp: 毫秒时间戳 (int)
      - datetime: UTC 时间 (datetime64[ns, UTC])
      - fear_greed_value: 指数值 (float)
      - price: 对应价格 (float)
    """
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/fear-greed-history"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    block = payload["data"][0]
    s_values = pd.Series(block["values"], dtype="float64")
    s_price  = pd.Series(block["price"], dtype="float64")
    s_time   = pd.Series(block["time_list"], dtype="int64")

    dt = pd.to_datetime(s_time, unit="s", utc=True)
    df = pd.DataFrame({
        "timestamp": (dt.view("int64") // 10**6).astype("int64"),
        "datetime": dt,
        "fear_greed_value": s_values,
        "price": s_price,
    }).sort_values("datetime").reset_index(drop=True)
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/fear_greed.csv", index=False)
    return df
# a = fetch_index_fear_greed()
# print(a.head())
def fetch_margin_long_short(
        symbol : str = "BTC",
        limit : int = 1000,
        interval: str = "4h",
        start_time: Optional[Union[str, int, float, datetime]] = None,
        end_time: Optional[Union[str, int, float, datetime]] = None
) -> pd.DataFrame:
        headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
        url = BASE + "/api/bitfinex-margin-long-short"
        params = {
            "symbol": symbol,
            "limit": limit,
            "interval": interval,
            "start_time": start_time,
            "end_time": end_time,
        }
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if str(payload["code"]) != "0":
            raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

        df = pd.DataFrame(payload["data"])
        # df["timestamp"] = pd.to_numeric(df["time"], errors="coerce")
        # df.drop(columns=["time"], inplace=True, errors="ignore")
        os.makedirs("data/cglass/index", exist_ok=True)
        df.to_csv(f"data/cglass/index/bitfinex_margin_long_short-{symbol}-{interval}.csv", index=False)
        return df
# a = fetch_margin_long_short(start_time=1641522717000)
# print(a.head())
def fetch_borrow_interest_rate(
        exchange: str = "Binance",
        symbol : str = "BTC",
        interval: str = "4h",
        limit : int = 1000,
        start_time: Optional[Union[str, int, float, datetime]] = None,
        end_time: Optional[Union[str, int, float, datetime]] = None
) -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/borrow-interest-rate/history"
    params = {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df["timestamp"] = pd.to_numeric(df["time"], errors="coerce")
    df.drop(columns=["time"], inplace=True, errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv(f"data/cglass/index/borrow_interest_rate-{exchange}-{symbol}-{interval}.csv", index=False)
    return df
# a = fetch_borrow_interest_rate()
# print(a.head())
def fetch_puell_multiple() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/puell-multiple"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/puell_multiple.csv", index=False)
    return df
# a = fetch_puell_multiple()
# print(a.head())
def fetch_stock_flow() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/stock-flow"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/stock_flow.csv", index=False)
    return df
# a = fetch_stock_flow()
# print(a.head())
def fetch_pi_cycle_indicator() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/pi-cycle-indicator"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/pi_cycle_indicator.csv", index=False)
    return df
# a = fetch_pi_cycle_indicator()
# print(a.head())
def fetch_golden_ratio_multiplier() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/golden-ratio-multiplier"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/golden_ratio_multiplier.csv", index=False)
    return df
# a = fetch_golden_ratio_multiplier()
# print(a.head())
def fetch_profitable_days() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin/profitable-days"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index/bitcoin", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin/profitable_days.csv", index=False)
    return df
# a = fetch_profitable_days()
# print(a.head())
def fetch_rainbow_chart() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin/rainbow-chart"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df["timestamp"] = df[11]
    df["lowest"] = df[1]
    df["first"]=df[2]
    df["second"] = df[3]
    df["third"] = df[4]
    df["fourth"] = df[5]
    df["fifth"] = df[6]
    df["sixth"] = df[7]
    df["seventh"] = df[8]
    df["eighth"] = df[9]
    df["ninth"] = df[10]
    df.drop(columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True, errors="ignore")
    df.drop(columns=[11], inplace=True, errors="ignore")
    df.drop(columns=[0], inplace=True, errors="ignore")
    os.makedirs("data/cglass/index/bitcoin", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin/rainbow_chart.csv", index=False)
    return df
# a = fetch_rainbow_chart()
# print(a.head())
def fetch_stableCoin_marketCap_history() -> pd.DataFrame:
    """
    读取 /api/index/stableCoin-marketCap-history 并返回三列：
      - timestamp: 来自 time_list（不改单位）
      - price:     来自 price_list（长度对齐 timestamp）
      - sum:       data_list 中每个时刻的字典按 value 求和（多稳定币合计）

    同时保存到 data/cglass/index/stableCoin-marketCap-history.csv
    """
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/stableCoin-marketCap-history"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"API error: code={payload.get('code')}, msg={payload.get('msg')}")

    # 兼容 data 外层 list / dict
    node = payload.get("data", {})
    if isinstance(node, list) and len(node) > 0:
        node = node[0]

    data_list = node.get("data_list") or node.get("datalist") or []
    price_list = node.get("price_list") or node.get("pricelist") or []
    time_list  = node.get("time_list")  or node.get("timelist")  or []

    if not isinstance(time_list, (list, tuple)) or len(time_list) == 0:
        raise ValueError("time_list 为空或格式不正确")

    T = len(time_list)

    # ===== 1) 计算 sum：逐时刻把稳定币字典的所有 value 相加 =====
    sums = []
    if isinstance(data_list, (list, tuple)) and len(data_list) > 0:
        # data_list 期望为长度≈T 的 list[dict]
        for i in range(T):
            if i < len(data_list) and isinstance(data_list[i], dict):
                # 将字典中可转换为 float 的值相加，忽略 None/无法转换的值
                total = 0.0
                any_val = False
                for v in data_list[i].values():
                    try:
                        if v is not None:
                            total += float(v)
                            any_val = True
                    except Exception:
                        pass
                sums.append(total if any_val else np.nan)
            else:
                sums.append(np.nan)
    else:
        # data_list 缺失或格式异常，返回全 NaN
        sums = [np.nan] * T

    sum_series = pd.Series(sums, dtype="float64")

    # ===== 2) 处理 price_list -> price：对齐长度 =====
    price_series = pd.to_numeric(pd.Series(price_list, dtype="float64"), errors="coerce")
    if len(price_series) < T:
        # 价格不足则补齐 NaN
        price_series = price_series.reindex(range(T))
    else:
        price_series = price_series.iloc[:T]

    # ===== 3) 组装结果并保存 =====
    df = pd.DataFrame({
        "timestamp": pd.to_numeric(pd.Series(time_list[:T]), errors="coerce"),
        "price": price_series.values,
        "total": sum_series.values,
    })
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/stableCoin-marketCap-history.csv", index=False)
    return df
# a = fetch_stableCoin_marketCap_history()
# print(a.head())
def fetch_bubble_index() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin/bubble-index"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    df["timestamp"]  = pd.to_datetime(df["date_string"], utc=True, errors="coerce").astype("int64") // 10**6
    os.makedirs("data/cglass/index/bitcoin", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin/bubble_index.csv", index=False)
    return df
# a = fetch_bubble_index()
# print(a.head())
def fetch_altcoin_season() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/altcoin-season"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index/altcoin", exist_ok=True)
    df.to_csv("data/cglass/index/altcoin-season.csv", index=False)
    return df
# a = fetch_altcoin_season()
# print(a.head())
def fetch_bitcoin_sth_sopr() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-sth-sopr"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-sth_sopr.csv", index=False)
    return df
# a = fetch_bitcoin_sth_sopr()
# print(a.head())
def fetch_bitcoin_lth_sopr()-> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-lth-sopr"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-lth_sopr.csv", index=False)
    return df
# a = fetch_bitcoin_lth_sopr()
# print(a.head())
def fetch_bitcoin_sth_realized_price() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-sth-realized-price"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-sth_realized_price.csv", index=False)
    return df
# a = fetch_bitcoin_sth_realized_price()
# print(a.head())
def fetch_bitcoin_lth_realized_price() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-lth-realized-price"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-lth_realized_price.csv", index=False)
    return df
# a = fetch_bitcoin_lth_realized_price()
# print(a.head())
def fetch_bitcoin_short_term_holder_supply() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-short-term-holder-supply"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-short_term_holder_supply.csv", index=False)
    return df
# a = fetch_bitcoin_short_term_holder_supply()
# print(a.head())
def fetch_bitcoin_long_term_holder_supply() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-long-term-holder-supply"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-long_term_holder_supply.csv", index=False)
    return df
# a = fetch_bitcoin_long_term_holder_supply()
# print(a.head())
def fetch_bitcoin_rhodl_ratio() ->pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-rhodl-ratio"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-rhodl_ratio.csv", index=False)
    return df
# a = fetch_bitcoin_rhodl_ratio()
# print(a.head())
def fetch_bitcoin_new_addresses() ->pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-new-addresses"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-new_addresses.csv", index=False)
    return df
# a = fetch_bitcoin_new_addresses()
# print(a.head())
def fetch_bitcoin_active_addresses() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-active-addresses"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-active_addresses.csv", index=False)
    return df
# a = fetch_bitcoin_active_addresses()
# print(a.head())
def fetch_bitcoin_reserve_risk() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-reserve-risk"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-reserve_risk.csv", index=False)
    return df
# a = fetch_bitcoin_reserve_risk()
# print(a.head())
def fetch_bitcoin_net_unrealized_profit_loss() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-net-unrealized-profit-loss"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-net_unrealized_profit_loss.csv", index=False)
    return df
# a = fetch_bitcoin_net_unrealized_profit_loss()
# print(a.tail())
def fetch_bitcoin_correlation() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-correlation"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-correlation.csv", index=False)
    return df
# a = fetch_bitcoin_correlation()
# print(a.head())
def fetch_bitcoin_macro_oscillator() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-macro-oscillator"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-macro_oscillator.csv", index=False)
    return df
# a = fetch_bitcoin_macro_oscillator()
# print(a.head())
def fetch_bitcoin_vs_global_m2_growth() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-vs-global-m2-growth"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-vs-global-m2-growth.csv", index=False)
    return df
# a = fetch_bitcoin_vs_global_m2_growth()
# print(a.head())
def fetch_bitcoin_vs_us_m2_growth() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-vs-us-m2-growth"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-vs-us-m2-growth.csv", index=False)
    return df
# a = fetch_bitcoin_vs_us_m2_growth()
# print(a.head())
def fetch_bitcoin_dominance() -> pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/index/bitcoin-dominance"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/index", exist_ok=True)
    df.to_csv("data/cglass/index/bitcoin-dominance.csv", index=False)
    return df
# a = fetch_bitcoin_dominance()
# print(a.head())
def fetch_futures_basis(
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        limit : int = 1000,
        start_time : Optional[int] = None,
        end_time : Optional[int] = None
    )   ->pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/futures/basis/history"

    params = {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df["timestamp"] = df["time"]
    df.drop(columns=["time"], inplace=True, errors="ignore")
    os.makedirs("data/cglass/futures", exist_ok=True)
    df.to_csv(f"data/cglass/futures/futures-basis-{exchange}-{symbol}-{interval}.csv", index=False)
    return df
# a = fetch_futures_basis()
# print(a.head())
def fetch_whale_index(
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        limit : int = 1000,
        start_time : Optional[int] = None,
        end_time : Optional[int] = None
    )   ->pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/futures/whale-index/history"

    params = {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    df["timestamp"] = df["time"]
    df.drop(columns=["time"], inplace=True, errors="ignore")
    os.makedirs("data/cglass/futures", exist_ok=True)
    df.to_csv(f"data/cglass/futures/whale-index-{exchange}-{symbol}-{interval}.csv", index=False)
    return df
# a = fetch_whale_index()
# print(a.head())
def fetch_cgdi_index(
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        limit : int = 1000,
        start_time : Optional[int] = None,
        end_time : Optional[int] = None
    )   ->pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + f"/api/futures/cgdi-index/history"

    params = {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    df["timestamp"] = df["time"]
    df.drop(columns=["time"], inplace=True, errors="ignore")
    os.makedirs("data/cglass/futures", exist_ok=True)
    df.to_csv(f"data/cglass/futures/futures-cgdi-{exchange}-{symbol}-{interval}.csv", index=False)
    return df
# a = fetch_cgdi_index()
# print(a.head())
def fetch_cdri_index(
        exchange: str = "Binance",
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        limit : int = 1000,
        start_time : Optional[int] = None,
        end_time : Optional[int] = None
    )   ->pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + f"/api/futures/cdri-index/history"

    params = {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "start_time": start_time,
        "end_time": end_time
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    df["timestamp"] = df["time"]
    df.drop(columns=["time"], inplace=True, errors="ignore")
    os.makedirs("data/cglass/futures", exist_ok=True)
    df.to_csv(f"data/cglass/futures/futures-cdri-{exchange}-{symbol}-{interval}.csv", index=False)
    return df
# a = fetch_cdri_index()
# print(a.head())
def fetch_etf_btc()->pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/etf/bitcoin/net-assets/history"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/futures", exist_ok=True)
    df.to_csv("data/cglass/futures/futures-etf-btc.csv", index=False)
    return df
# a = fetch_etf_btc()
# print(a.head())
def fetch_etf_eth()->pd.DataFrame:
    headers = {"accept": "application/json", "CG-API-KEY": API_KEY}
    url = BASE + "/api/etf/ethereum/net-assets/history"

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if str(payload["code"]) != "0":
        raise RuntimeError(f"API error: code={payload['code']}, msg={payload.get('msg')}")

    df = pd.DataFrame(payload["data"])
    df = df.drop(columns=["price"], errors="ignore")
    os.makedirs("data/cglass/futures", exist_ok=True)
    df.to_csv("data/cglass/futures/futures-etf-eth.csv", index=False)
    return df
# a = fetch_etf_eth()
# print(a.head())
FUNCTION_NAMES = [
    "_infer_default_window",
    "_ensure_numeric",
    "_roll_zscore",
    "_roll_robust",
    "_roll_minmax",
    "_to_ms",
    "fetch_funding_rate",
    "accum_as_of",
    "long_short_account_ratio",
    "fetch_oi",
    "taker_buy_sell_volume",
    "premium_index",
    "fetch_index_AHR999",
    "fetch_index_fear_greed",
    "fetch_margin_long_short",
    "fetch_borrow_interest_rate",
    "fetch_puell_multiple",
    "fetch_stock_flow",
    "fetch_pi_cycle_indicator",
    "fetch_golden_ratio_multiplier",
    "fetch_profitable_days",
    "fetch_rainbow_chart",
    "fetch_stableCoin_marketCap_history",
    "fetch_bubble_index",
    "fetch_altcoin_season",
    "fetch_bitcoin_sth_sopr",
    "fetch_bitcoin_lth_sopr",
    "fetch_bitcoin_sth_realized_price",
    "fetch_bitcoin_lth_realized_price",
    "fetch_bitcoin_short_term_holder_supply",
    "fetch_bitcoin_long_term_holder_supply",
    "fetch_bitcoin_rhodl_ratio",
    "fetch_bitcoin_new_addresses",
    "fetch_bitcoin_active_addresses",
    "fetch_bitcoin_reserve_risk",
    "fetch_bitcoin_net_unrealized_profit_loss",
    "fetch_bitcoin_correlation",
    "fetch_bitcoin_macro_oscillator",
    "fetch_bitcoin_vs_global_m2_growth",
    "fetch_bitcoin_vs_us_m2_growth",
    "fetch_bitcoin_dominance",
    "fetch_futures_basis",
    "fetch_whale_index",
    "fetch_cgdi_index",
    "fetch_cdri_index",
    "fetch_etf_btc", 
    "fetch_etf_eth",
]


    