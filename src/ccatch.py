import ccxt
import pandas as pd
import time
import os
from datetime import datetime

# ===== 可配置参数 =====
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
timeframe = '1h'  # 可切换为 '1h'、'4h'、'1d'
since_str = '2018-01-01T00:00:00Z'
limit = 1000
# python src/ccatch.py
# ===== 正确的路径配置：退回到项目根目录再定位 data =====
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
crypto_dir = os.path.join(project_root, 'data', 'crypto', timeframe)
os.makedirs(crypto_dir, exist_ok=True)

# ===== 初始化 Binance 接口 =====
exchange = ccxt.binance()
exchange.load_markets()

# ===== 拉取数据函数 =====
def fetch_ohlcv_all(symbol, timeframe, since_str, limit):
    print(f"\n📥 正在获取 {symbol} 的 {timeframe} 数据...")
    since = exchange.parse8601(since_str)
    all_data = []

    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_data += ohlcv
        since = ohlcv[-1][0] + 1
        print(f"✅ 获取至：{datetime.utcfromtimestamp(since / 1000)}，共 {len(all_data)} 条")
        time.sleep(exchange.rateLimit / 1000)
        if since >= exchange.milliseconds():
            break
    return all_data

# ===== 主流程：批量拉取并保存为 CSV =====
def main():
    for symbol in symbols:
        data = fetch_ohlcv_all(symbol, timeframe, since_str, limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

        # === ✅ 添加标志字段 ===
        df['symbol'] = symbol.replace('/', '_')  # eg. BTC_USDT
        df['period'] = timeframe                 # eg. 1h, 4h, 1d

        # === 保存路径 ===
        filename = f"{df['symbol'].iloc[0]}_{timeframe}_all.csv"
        filepath = os.path.join(crypto_dir, filename)

        df.to_csv(filepath, index=False)
        print(f"💾 已保存到：{filepath}")

if __name__ == "__main__":
    main()
