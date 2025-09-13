# save as: src/ccatch.py
import os
import time
import ccxt
import pandas as pd
from datetime import datetime

# ====================== 配置区 ======================
BASE_SYMBOLS   = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ADA/USDT']
TIMEFRAMES     = ['1h', '4h', '1d']
SINCE_STR      = '2020-10-01T00:00:00Z'
LIMIT          = 1500
CHUNK_FLUSH    = 20000
PRICE_MODE     = 'trade'   # 'trade' | 'mark' | 'index'
EXCHANGE_TYPE  = 'usdm'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_ROOT     = os.path.join(PROJECT_ROOT, 'data', 'crypto')

# ====================== 交易所初始化 ======================
def make_exchange():
    ex = ccxt.binanceusdm({'enableRateLimit': True})
    if not ex.markets:
        ex.load_markets()
    return ex

exchange = make_exchange()

# ====================== 工具函数 ======================
def tf_ms(tf: str) -> int:
    return int(exchange.parse_timeframe(tf) * 1000)

def iso8601_to_ms(s: str) -> int:
    return exchange.parse8601(s)

def now_ms() -> int:
    return exchange.milliseconds()

def pick_perp_symbol(std_symbol: str) -> str | None:
    """
    从标准现货名 'ETH/USDT' 里挑选永续 'ETH/USDT:USDT'（优先 USDT 结算）
    """
    cands = [
        m for m in exchange.markets.values()
        if (m.get('base') and m.get('quote')
            and f"{m['base']}/{m['quote']}" == std_symbol
            and m.get('type') == 'swap')
    ]
    if cands:
        # settle == 'USDT' 优先
        cands.sort(key=lambda m: (m.get('settle') == 'USDT') is False)
        return cands[0]['symbol']
    if std_symbol in exchange.symbols and exchange.market(std_symbol).get('type') == 'swap':
        return std_symbol
    return None

def safe_fetch_ohlcv(symbol, timeframe, since, limit, params=None):
    backoff = 1.0
    while True:
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit, params=params)
        except ccxt.RateLimitExceeded:
            sleep = max(exchange.rateLimit / 1000.0, backoff)
            print(f"⏳ Rate limited, sleep {sleep:.2f}s"); time.sleep(sleep)
            backoff = min(backoff * 2.0, 60.0)
        except ccxt.DDoSProtection as e:
            print(f"🛑 DDoSProtection: {e}. Cooling down 60s"); time.sleep(60)
            backoff = min(backoff * 2.0, 120.0)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            print(f"🌐 Network/Exchange error: {e}. Retry in {backoff:.1f}s"); time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)

def _ensure_csv_with_header(path):
    if not os.path.exists(path):
        cols = ['timestamp','open','high','low','close','volume','datetime','symbol','period']
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        print(f"📝 初始化 CSV：{os.path.abspath(path)}")

def _safe_name(sym: str) -> str:
    """
    统一命名：去掉结算后缀并替换分隔符
    'ETH/USDT:USDT'  -> 'ETH_USDT'
    'BTC/USD:USD'    -> 'BTC_USD'
    仅用于 文件名 和 CSV 内 'symbol' 列，避免出现 '..._USDT_USDT'
    """
    core = sym.split(':', 1)[0]     # 去掉 :USDT / :USD / :USDC 等
    return core.replace('/', '_')

def _legacy_safe_name(sym: str) -> str:
    """
    旧版的“全替换”写法（会得到 ETH_USDT_USDT）
    用于自动迁移已存在的历史文件名
    """
    return sym.replace('/', '_').replace(':', '_')

def _maybe_migrate_legacy_file(out_dir: str, sym: str, tf: str) -> str:
    """
    若存在老文件名（*_USDT_USDT_*.csv），自动重命名成新规范（无重复 USDT）
    """
    new_safe = _safe_name(sym)
    old_safe = _legacy_safe_name(sym)
    new_path = os.path.join(out_dir, f"{new_safe}_{tf}_all.csv")
    old_path = os.path.join(out_dir, f"{old_safe}_{tf}_all.csv")
    if os.path.exists(old_path) and not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print(f"♻️ 重命名旧文件：{old_path} → {new_path}")
    return new_path

def flush_append_csv(rows, dst_csv, symbol, timeframe):
    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
    # 统一用 UTC 时间戳字符串，去掉 tz 信息以便后续处理
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    # 列里的 symbol 用“去重后的安全名”
    df['symbol']   = _safe_name(symbol)
    df['period']   = timeframe
    write_header   = not os.path.exists(dst_csv)
    df.to_csv(dst_csv, mode='a', header=write_header, index=False)

# ====================== 交错抓取主循环 ======================
def catch_main():
    base_since = iso8601_to_ms(SINCE_STR)

    tasks = []
    for tf in TIMEFRAMES:
        out_dir = os.path.join(OUT_ROOT, tf)
        os.makedirs(out_dir, exist_ok=True)
        step = tf_ms(tf)

        for s in BASE_SYMBOLS:
            perp = pick_perp_symbol(s)
            if not perp:
                print(f"⚠️ 跳过 {s}（未找到永续市场）"); continue

            # 新路径（含自动迁移旧文件名）
            dst = _maybe_migrate_legacy_file(out_dir, perp, tf)
            _ensure_csv_with_header(dst)
            print(f"➡️ 输出路径：{os.path.abspath(dst)}")

            since_ms = base_since
            # 断点续抓
            if os.path.exists(dst):
                try:
                    last = pd.read_csv(dst, usecols=['timestamp']).tail(1)
                    if len(last):
                        since_ms = int(last['timestamp'].iloc[0]) + 1
                        print(f"🔁 {perp} {tf} resume from {datetime.utcfromtimestamp(since_ms/1000)} UTC")
                except Exception as e:
                    print(f"⚠️ 读取 {dst} 失败，忽略断点：{e}")

            tasks.append({
                "symbol": perp, "timeframe": tf, "since": since_ms,
                "buf": [], "prev_last": None, "path": dst, "step": step
            })

    if not tasks:
        print("❌ 没有可抓取任务"); return

    params = {} if PRICE_MODE == 'trade' else {'price': PRICE_MODE}

    try:
        active = True
        while active:
            active = False
            for t in tasks:
                symbol, tf, since_ms = t["symbol"], t["timeframe"], t["since"]
                if since_ms >= now_ms():
                    continue
                active = True

                ohlcv = safe_fetch_ohlcv(symbol, tf, since_ms, LIMIT, params=params)
                if not ohlcv:
                    t["since"] += t["step"]
                    continue

                last_ts = ohlcv[-1][0]
                if t["prev_last"] is not None and last_ts <= t["prev_last"]:
                    time.sleep(max(exchange.rateLimit, 250) / 1000.0)
                    t["since"] += t["step"]
                    continue

                t["buf"].extend(ohlcv)
                t["since"]     = last_ts + 1
                t["prev_last"] = last_ts

                print(f"✅ {symbol} {tf} → {datetime.utcfromtimestamp(t['since']/1000)} UTC (+{len(ohlcv)}), buf={len(t['buf'])})")

                if len(t["buf"]) >= CHUNK_FLUSH:
                    flush_append_csv(t["buf"], t["path"], symbol, tf)
                    t["buf"].clear()

    finally:
        for t in tasks:
            if t["buf"]:
                flush_append_csv(t["buf"], t["path"], t["symbol"], t["timeframe"])
                t["buf"].clear()
                print(f"💾 保存完成：{t['path']}")

if __name__ == "__main__":
    catch_main()
