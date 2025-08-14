# save as: ingest_provider.py
import os, time, math, json, requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# =============== 配置区（按你的平台改） ===============
BASE_URL = "https://<your-provider-domain>"          # TODO: 改成你的域名
API_KEY  = os.getenv("PROVIDER_API_KEY", "<APIKEY>") # TODO: 放环境变量更安全

HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# 常用接口（按你平台文档核对路径）
ENDPOINTS = {
    # 快频（1h/4h）
    "ohlc": "/api/price/ohlc-history",
    "oi_pair": "/api/futures/openInterest/ohlc-history",
    "funding": "/api/futures/fundingRate/ohlc-history",
    "funding_oiw": "/api/futures/fundingRate/oi-weight-ohlc-history",
    "funding_volw": "/api/futures/fundingRate/vol-weight-ohlc-history",
    "ls_all": "/api/futures/global-long-short-account-ratio/history",
    "ls_top": "/api/futures/top-long-short-account-ratio/history",
    "taker_pair": "/api/futures/taker-buy-sell-volume/history",
    "taker_coin": "/api/futures/aggregated-taker-buy-sell-volume/history",
    "liq_pair": "/api/futures/liquidation/history",
    "liq_coin": "/api/futures/liquidation/aggregated-history",

    # 慢频（1d）
    "etf_btc_flow": "/api/etf/bitcoin/flow-history",
    "etf_btc_aum": "/api/etf/bitcoin/net-assets/history",
    "etf_eth_flow": "/api/etf/ethereum/flow-history",
    "etf_eth_aum": "/api/etf/ethereum/net-assets-history",
    "ex_balance_chart": "/api/exchange/balance/chart",
    "erc20_tx": "/api/exchange/chain/tx/list",
    "fear_greed": "/api/index/fear-greed-history",
    "stablecoin_mcap": "/api/index/stableCoin-marketCap-history",
    "cb_premium": "/api/coinbase-premium-index",
    "basis": "/api/futures/basis/history",
}


SYMBOLS = ["BTCUSDT", "ETHUSDT"]   # 你在平台的交易对命名；如是 BTC-USD 请对应改
PERIODS_FAST = ["1h", "4h"]
PERIOD_SLOW  = "1d"
DAYS = 360  # 初创版历史窗

# 字段映射（把平台返回的字段映射到你统一的长表字段名）
MAP_PRICE = {"ts":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}
MAP_OI    = {"ts":"timestamp","oi":"oi_total"}  # 如果有 usdt/coin margin 可再扩充
MAP_FUND  = {"ts":"timestamp","v":"funding_rate"}
MAP_LS_ALL= {"ts":"timestamp","v":"long_short_ratio_all"}
MAP_LS_TOP= {"ts":"timestamp","v":"long_short_ratio_top"}
MAP_ETF_HOLD = {"ts":"timestamp","v":"etf_holdings"}
MAP_ETF_FLOW = {"ts":"timestamp","v":"etf_net_flow"}
MAP_EX_IN   = {"ts":"timestamp","v":"ex_inflow"}
MAP_EX_OUT  = {"ts":"timestamp","v":"ex_outflow"}

def _ts_to_utc_seconds(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def _since_until(days: int):
    until = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    since = until - timedelta(days=days)
    return _ts_to_utc_seconds(since)*1000, _ts_to_utc_seconds(until)*1000  # 某些平台毫秒

def _get(url, params, retry=3, sleep=0.25):
    for k in range(retry):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(sleep*(k+1))
    raise RuntimeError(f"GET failed: {url} {params} -> {r.status_code} {r.text[:200]}")

def _norm_df(raw, mapping: dict, symbol: str, period: str):
    # raw: list[dict] 或 dict 包含 "data"
    data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    if not data:
        return pd.DataFrame(columns=["timestamp","symbol","period",*mapping.values()])
    df = pd.DataFrame(data)
    # 统一字段
    rename = {k:v for k,v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename)
    # 统一时间戳为秒级 UTC
    if "timestamp" not in df.columns:
        # 常见名称兼容
        for cand in ["ts","time","timestampMs","T"]:
            if cand in df.columns:
                df["timestamp"] = df[cand]
                break
    # 毫秒->秒
    df["timestamp"] = (df["timestamp"] / (1000 if df["timestamp"].max()>1e12 else 1)).astype("int64")
    df = df[["timestamp", *[c for c in mapping.values() if c!="timestamp"]]]
    df["symbol"] = symbol
    df["period"] = period
    # 排序去重
    df = df.sort_values("timestamp").drop_duplicates(["timestamp","symbol","period"])
    return df

def fetch_series(endpoint_key, symbol, period, days=DAYS, **extra):
    since, until = _since_until(days)
    url = BASE_URL + ENDPOINTS[endpoint_key]
    params = {"symbol": symbol, "period": period, "since": since, "until": until}
    params.update(extra)
    return _get(url, params=params)

def fetch_block(symbols=SYMBOLS, periods=PERIODS_FAST):
    frames = []

    for s in symbols:
        # 价格
        for p in periods:
            raw = fetch_series("ohlc", s, p)
            frames.append(_norm_df(raw, MAP_PRICE, s, p))
        # OI
        for p in periods:
            raw = fetch_series("oi_pair", s, p)
            frames.append(_norm_df(raw, MAP_OI, s, p))
        # funding
        for p in periods:
            raw = fetch_series("funding", s, p)
            frames.append(_norm_df(raw, MAP_FUND, s, p))
        # 多空比
        for p in periods:
            frames.append(_norm_df(fetch_series("ls_all", s, p), MAP_LS_ALL, s, p))
            frames.append(_norm_df(fetch_series("ls_top", s, p), MAP_LS_TOP, s, p))
    return pd.concat(frames, ignore_index=True)

def fetch_slow_macro(symbols=SYMBOLS):
    # 慢频（1d）：ETF & 链上净流
    frames = []
    for s in symbols:
        frames.append(_norm_df(fetch_series("etf_hold", s, PERIOD_SLOW), MAP_ETF_HOLD, s, PERIOD_SLOW))
        frames.append(_norm_df(fetch_series("etf_flow", s, PERIOD_SLOW), MAP_ETF_FLOW, s, PERIOD_SLOW))
        frames.append(_norm_df(fetch_series("ex_inflow", s, PERIOD_SLOW), MAP_EX_IN, s, PERIOD_SLOW))
        frames.append(_norm_df(fetch_series("ex_outflow", s, PERIOD_SLOW), MAP_EX_OUT, s, PERIOD_SLOW))
    df = pd.concat(frames, ignore_index=True)

    # 计算净流（如果只拿到了 in/out）
    if {"ex_inflow","ex_outflow"}.issubset(df.columns):
        df["ex_netflow"] = df["ex_inflow"].fillna(0) - df["ex_outflow"].fillna(0)
    return df

def pivot_long(df):
    """把不同指标叠成行已是长表，这里只做类型压缩与排序"""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["symbol","period","timestamp"]).reset_index(drop=True)
    return df

def downcast(df):
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        if df[c].dtype == "int64" and c != "timestamp":
            df[c] = df[c].astype("int32")
    return df

def broadcast_and_ffill(macro_df, fast_df):
    """把 1d 的 macro 字段下放到 4h/1h（同一 symbol）"""
    out = []
    for s in fast_df["symbol"].unique():
        base = fast_df[fast_df["symbol"]==s].copy()
        macro = macro_df[macro_df["symbol"]==s].copy()
        if macro.empty:
            out.append(base); continue
        # 先 pivot 宏观字段到列
        cols = [c for c in macro.columns if c not in ["timestamp","symbol","period"]]
        macro_wide = macro.pivot_table(index="timestamp", values=cols, aggfunc="last").sort_index()
        # 合并到快频时间戳
        base = base.sort_values("timestamp")
        base = base.merge(macro_wide, left_on="timestamp", right_index=True, how="left")
        base[cols] = base[cols].ffill()  # 下放 & 前向填充
        out.append(base)
    return pd.concat(out, ignore_index=True)

def main(out_path="data/merged/raw_market_360d.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print("↻ Fetching fast (1h/4h) blocks ...")
    fast = fetch_block()
    print(f"fast: {fast.shape}")
    print("↻ Fetching slow (1d) macro ...")
    slow = fetch_slow_macro()
    print(f"slow: {slow.shape}")

    print("↻ Merge & downcast ...")
    fast = pivot_long(fast)
    slow = pivot_long(slow)

    merged = broadcast_and_ffill(slow, fast)
    merged = downcast(merged)

    merged = merged.sort_values(["timestamp","symbol","period"]).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f"✅ saved -> {out_path}")

if __name__ == "__main__":
    main()
