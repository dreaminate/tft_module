# -*- coding: utf-8 -*-
"""
apied.py — 批量抓取衍生品/链上指标的入口脚本

用法示例：
  # 1) 传入“日期字符串”，自动转成毫秒时间戳（UTC 00:00:00）
  python src/apied.py --start 2020-10-02 --end 2025-12-18

  # 2) 直接传整数（10位秒或13位毫秒都行）
  python src/apied.py --start 1601596800 --end 1766275200000

  # 3) 自定义币种/间隔/限速
  python src/apied.py --symbols BTC,ETH,SOL,BNB,ADA --interval 1d --pause 1.0

支持的时间输入：
  - 13位毫秒：1755043200000
  - 10位秒：  1601510400
  - 日期：    2020-10-02 / 20201002 / 2020-10-02T08:30:00 / 2020-10-02 08:30:00
  - 关键字：  now（当前UTC时间）
"""
# python 3.7+
# python src/apied.py --start 2020-10-02 --end 2025-12-18
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone

# 按你的原调用保持一致：直接从 cgapi 引入函数
from cgapi import *  # noqa: F401,F403


# ===================== 时间解析工具（统一转为毫秒） =====================
def _to_ms(dt: datetime) -> int:
    # 统一使用 UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def parse_time_input(x) -> int:
    """
    将多种格式的时间输入统一解析为 13 位毫秒时间戳（UTC）。
    支持：13位毫秒、10位秒、YYYY-MM-DD、YYYYMMDD、ISO8601（含空格/T，支持Z/+00:00）、'now'
    """
    if x is None:
        return None

    # 数字类型或纯数字字符串
    if isinstance(x, (int, float)):
        v = int(x)
        return v if v >= 10**12 else v * 1000

    s = str(x).strip()
    if s.lower() == "now":
        return _to_ms(datetime.now(timezone.utc))

    # 13位毫秒 / 10位秒
    if re.fullmatch(r"\d{13}", s):
        return int(s)
    if re.fullmatch(r"\d{10}", s):
        return int(s) * 1000

    # YYYYMMDD
    if re.fullmatch(r"\d{8}", s):
        dt = datetime.strptime(s, "%Y%m%d")
        return _to_ms(dt)

    # 统一处理 ISO 格式；支持带空格或'T'；支持Z
    try:
        s_iso = s.replace("Z", "+00:00")  # 兼容结尾Z
        # 仅日期（YYYY-MM-DD）=> 当天 00:00:00
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s_iso):
            dt = datetime.strptime(s_iso, "%Y-%m-%d")
        else:
            dt = datetime.fromisoformat(s_iso)
        return _to_ms(dt)
    except Exception:
        raise ValueError(
            f"无法解析时间输入：{x!r}。请使用 13位毫秒 / 10位秒 / 'YYYY-MM-DD' / 'YYYYMMDD' / ISO8601 / 'now'."
        )


def ms_to_iso(ms: int) -> str:
    return datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S UTC")


# ===================== 安全调用 + 限速工具 =====================
def safe_call(fn, *args, pause: float = 1.0, **kwargs):
    """包装一次函数调用，异常不中断，并在成功或失败后 sleep 一下做限速。"""
    name = getattr(fn, "__name__", str(fn))
    try:
        print(f"→ {name}({kwargs if kwargs else ''})")
        ret = fn(*args, **kwargs)
        time.sleep(pause)
        return ret
    except Exception as e:
        print(f"[WARN] {name} 调用失败：{e}")
        time.sleep(pause)
        return None


# ===================== 主流程 =====================
def apied_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTC,ETH,SOL,BNB,ADA",
                        help="逗号分隔，如：BTC,ETH,SOL,BNB,ADA")
    parser.add_argument("--interval", default="1d", help="时间粒度，默认 1d")
    parser.add_argument("--limit", type=int, default=4500, help="每次拉取上限，默认 4500")
    parser.add_argument("--pause", type=float, default=1.0, help="每次调用后的 sleep 秒数，默认 1.0")
    # 起止时间：支持任意受支持格式（见上）
    parser.add_argument("--start", default="2020-10-01", help="起始时间（如 2020-10-02、13位毫秒、10位秒、now 等）")
    parser.add_argument("--end", default="now", help="终止时间（同上），注意部分接口可能是开区间/闭区间由服务端决定")

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    interval = args.interval
    limit = args.limit
    pause = args.pause

    start_ms = parse_time_input(args.start)
    end_ms = parse_time_input(args.end)

    print("==========================================")
    print(f"Symbols:   {symbols}")
    print(f"Interval:  {interval}    Limit: {limit}    Pause: {pause}s")
    print(f"Start:     {args.start}  → {start_ms}  ({ms_to_iso(start_ms)})")
    print(f"End:       {args.end}    → {end_ms}    ({ms_to_iso(end_ms)})")
    print("==========================================\n")

    # ====== 按币种逐一拉取（与你原代码一致）======
    for symbol in symbols:
        pair = f"{symbol}USDT"  # 你原本在这些接口里使用的是无斜杠格式
        print(f"\n===== {symbol} =====")

        safe_call(fetch_funding_rate,
                  symbol, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

        safe_call(fetch_oi,
                  symbol, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

        # safe_call(fetch_margin_long_short,
        #           symbol=symbol, interval=interval, limit=limit,
        #           start_time=start_ms, end_time=end_ms, pause=pause)

        safe_call(fetch_borrow_interest_rate,
                  symbol=symbol, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

        safe_call(taker_buy_sell_volume,
                  symbol=pair, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

        safe_call(long_short_account_ratio,
                  symbol=pair, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

        safe_call(fetch_futures_basis,
                  symbol=pair, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

        safe_call(fetch_whale_index,
                  symbol=pair, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

        safe_call(fetch_cgdi_index,
                  symbol=pair, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

        safe_call(fetch_cdri_index,
                  symbol=pair, interval=interval, limit=limit,
                  start_time=start_ms, end_time=end_ms, pause=pause)

    # ====== 与币种无关或单独拉取的指标 ======
    print("\n===== Global / BTC-only =====")
    safe_call(fetch_altcoin_season, pause=pause)
    safe_call(fetch_bitcoin_active_addresses, pause=pause)
    safe_call(fetch_bitcoin_correlation, pause=pause)
    safe_call(fetch_bitcoin_dominance, pause=pause)
    safe_call(fetch_bitcoin_long_term_holder_supply, pause=pause)
    safe_call(fetch_index_AHR999, pause=pause)
    safe_call(fetch_bitcoin_lth_realized_price, pause=pause)
    safe_call(fetch_bitcoin_macro_oscillator, pause=pause)
    safe_call(fetch_bitcoin_sth_realized_price, pause=pause)
    safe_call(fetch_bitcoin_sth_sopr, pause=pause)
    safe_call(fetch_bitcoin_lth_sopr, pause=pause)
    safe_call(fetch_stock_flow, pause=pause)
    safe_call(fetch_golden_ratio_multiplier, pause=pause)
    safe_call(fetch_pi_cycle_indicator, pause=pause)
    # safe_call(fetch_profitable_days, pause=pause)
    safe_call(fetch_bitcoin_rhodl_ratio, pause=pause)
    safe_call(fetch_etf_btc, pause=pause)
    safe_call(fetch_etf_eth, pause=pause)
    safe_call(fetch_bitcoin_net_unrealized_profit_loss, pause=pause)
    safe_call(fetch_bitcoin_reserve_risk, pause=pause)
    safe_call(fetch_bitcoin_vs_global_m2_growth, pause=pause)
    safe_call(fetch_bitcoin_vs_us_m2_growth, pause=pause)
    safe_call(fetch_stableCoin_marketCap_history, pause=pause)
    safe_call(fetch_rainbow_chart, pause=pause)
    safe_call(fetch_bitcoin_short_term_holder_supply, pause=pause)

    # 这个接口你原本需要起止时间参数，这里也串上
    safe_call(premium_index, start_time=start_ms, end_time=end_ms, pause=pause)

    safe_call(fetch_bubble_index, pause=pause)
    safe_call(fetch_puell_multiple, pause=pause)
    safe_call(fetch_bitcoin_new_addresses, pause=pause)

    print("\n✅ 全部任务已触发完成。")


if __name__ == "__main__":
    apied_main()
