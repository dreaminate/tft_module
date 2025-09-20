from __future__ import annotations
import os
import argparse
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np


# --------------- 小工具 ---------------
def _ensure_int_ms(ts: pd.Series) -> pd.Series:
    s = pd.to_numeric(ts, errors="coerce")
    if s.dropna().astype(np.int64).astype(str).str.len().median() <= 10:
        s = s * 1000
    return s.astype("Int64")


def _read_csv_head(path: str, usecols: List[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False, usecols=usecols)
    except Exception:
        df = pd.read_csv(path, low_memory=False)
    return df


def _left_join_on_timestamp(base: pd.DataFrame, add: pd.DataFrame, on_cols: List[str]) -> pd.DataFrame:
    if add.empty:
        return base
    return base.merge(add, on=on_cols, how="left")


def _rename_cols(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    hit = {k: v for k, v in mapping.items() if k in df.columns}
    if hit:
        df = df.rename(columns=hit)
    return df


def _num_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _coin_from_symbol(symbol_usdt: str) -> str:
    # "BTC_USDT" -> "BTC"
    return symbol_usdt.split("_")[0]


def _pair_from_symbol(symbol_usdt: str) -> str:
    # "BTC_USDT" -> "BTCUSDT"
    a, b = symbol_usdt.split("_")
    return f"{a}{b}"


# --------------- 读取单币种基本面 ---------------
def load_symbol_fundamentals(
    symbol_usdt: str,
    include: Dict[str, bool] | None = None,
) -> pd.DataFrame:
    """
    汇总一个币种（如 BTC_USDT）的 1d 基本面到一个宽表（timestamp,symbol,若干列）。
    仅依赖本地 data/cglass 目录，存在哪个文件就并入哪个。
    """
    coin = _coin_from_symbol(symbol_usdt)         # BTC
    pair = _pair_from_symbol(symbol_usdt)         # BTCUSDT

    rows: List[pd.DataFrame] = []

    # 默认全开
    inc = {
        "funding_oi": True,
        "funding_vol": True,
        "oi": True,
        "ls_global": True,
        "ls_top_acc": True,
        "ls_top_pos": True,
        "basis": True,
        "whale": True,
        "taker": True,
        # 其他符号级指标
        "cgdi": True,
        "cdri": True,
        "borrow_interest": True,
    }
    if include:
        inc.update(include)

    # 1) 资金费率（OI加权 / 成交量加权）
    if inc.get("funding_oi", True):
        f_oi = f"data/cglass/futures/funding-rate/funding-rate-oi/Agg_{coin}_1d.csv"
        df = _read_csv_head(f_oi)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df[["timestamp", "close"]].copy()
            df = df.rename(columns={"close": "funding_rate_oi_close"})
            df["symbol"] = symbol_usdt
            rows.append(df)
    if inc.get("funding_vol", True):
        f_vol = f"data/cglass/futures/funding-rate/funding-rate-volume/Agg_{coin}_1d.csv"
        df = _read_csv_head(f_vol)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df[["timestamp", "close"]].copy()
            df = df.rename(columns={"close": "funding_rate_vol_close"})
            df["symbol"] = symbol_usdt
            rows.append(df)

    # 2) 未平仓量（OI 扩展特征）
    if inc.get("oi", True):
        f_oiagg = f"data/cglass/futures/open-interest/{symbol_usdt}_1d.csv"
        df = _read_csv_head(f_oiagg)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            keep = [c for c in ["oi_mean_oc", "oi_vol", "oi_mean_oc_z"] if c in df.columns]
            cols = ["timestamp", *keep]
            df = df[cols].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    # 3) 多空比（global / top account / top position）
    if inc.get("ls_global", True):
        ls_global = f"data/cglass/futures/long-short/global-long-short-account-ratio/Binance_{pair}_1d.csv"
        df = _read_csv_head(ls_global)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = _rename_cols(df, {
                "global_account_long_short_ratio": "long_short_ratio_ls_acc_global",
                "global_account_long_percent": "long_percent_ls_acc_global",
                "global_account_short_percent": "short_percent_ls_acc_global",
            })
            keep = [c for c in [
                "long_short_ratio_ls_acc_global",
                "long_percent_ls_acc_global",
                "short_percent_ls_acc_global",
            ] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    if inc.get("ls_top_acc", True):
        ls_top_acc = f"data/cglass/futures/long-short/top-long-short-account-ratio/Binance_{pair}_1d.csv"
        df = _read_csv_head(ls_top_acc)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = _rename_cols(df, {
                "top_account_long_short_ratio": "long_short_ratio_ls_acc_top",
                "top_account_long_percent": "long_percent_ls_acc_top",
                "top_account_short_percent": "short_percent_ls_acc_top",
            })
            keep = [c for c in [
                "long_short_ratio_ls_acc_top",
                "long_percent_ls_acc_top",
                "short_percent_ls_acc_top",
            ] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    if inc.get("ls_top_pos", True):
        ls_top_pos = f"data/cglass/futures/long-short/top-long-short-position-ratio/Binance_{pair}_1d.csv"
        df = _read_csv_head(ls_top_pos)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = _rename_cols(df, {
                "top_position_long_short_ratio": "long_short_ratio_ls_pos_top",
                "top_position_long_percent": "long_percent_ls_pos_top",
                "top_position_short_percent": "short_percent_ls_pos_top",
            })
            keep = [c for c in [
                "long_short_ratio_ls_pos_top",
                "long_percent_ls_pos_top",
                "short_percent_ls_pos_top",
            ] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    # 4) 期现基差
    if inc.get("basis", True):
        f_basis = f"data/cglass/futures/futures-basis-Binance-{pair}-1d.csv"
        df = _read_csv_head(f_basis)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = _rename_cols(df, {
                "open_basis": "open_basis_futures_basis",
                "close_basis": "close_basis_futures_basis",
                "open_change": "open_change_futures_basis",
                "close_change": "close_change_futures_basis",
            })
            keep = [c for c in [
                "open_basis_futures_basis", "close_basis_futures_basis",
                "open_change_futures_basis", "close_change_futures_basis",
            ] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    # 5) 鲸鱼指数
    if inc.get("whale", True):
        f_whale = f"data/cglass/futures/whale-index-Binance-{pair}-1d.csv"
        df = _read_csv_head(f_whale)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = _rename_cols(df, {"whale_index_value": "whale_index_value_whale_index"})
            keep = [c for c in ["whale_index_value_whale_index"] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    # 6) 吃单买卖量（现货）
    if inc.get("taker", True):
        f_taker = f"data/cglass/spot/taker-buy-sell-volume/{pair}_1d.csv"
        df = _read_csv_head(f_taker)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            keep = [c for c in [
                "taker_buy_volume_usd", "taker_sell_volume_usd", "taker_imbalance",
                "taker_buy_volume_usd_z", "taker_sell_volume_usd_z", "taker_imbalance_z",
            ] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    # CGDI 指数（期货动量/强弱）
    if inc.get("cgdi", True):
        f = f"data/cglass/futures/futures-cgdi-Binance-{pair}-1d.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df.get("timestamp", df.get("time"))).astype("int64")
            df = _rename_cols(df, {"cgdi_index_value": "cgdi_index_value_futures_cgdi"})
            keep = [c for c in ["cgdi_index_value_futures_cgdi"] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    # CDRI 指数（期货深度/多空差）
    if inc.get("cdri", True):
        f = f"data/cglass/futures/futures-cdri-Binance-{pair}-1d.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df.get("timestamp", df.get("time"))).astype("int64")
            df = _rename_cols(df, {"cdri_index_value": "cdri_index_value_futures_cdri"})
            keep = [c for c in ["cdri_index_value_futures_cdri"] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    # 借币利率（按币种/交易所）
    if inc.get("borrow_interest", True):
        f = f"data/cglass/index/borrow_interest_rate-Binance-{coin}-1d.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df.get("timestamp", df.get("time"))).astype("int64")
            df = _rename_cols(df, {"interest_rate": "interest_rate_borrow_interest_rate_binance"})
            keep = [c for c in ["interest_rate_borrow_interest_rate_binance"] if c in df.columns]
            df = df[["timestamp", *keep]].copy()
            df["symbol"] = symbol_usdt
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol"])  # 空壳

    out = rows[0]
    for d in rows[1:]:
        out = out.merge(d, on=["timestamp", "symbol"], how="outer")
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


# --------------- 读取全局/比特币全市场指标 ---------------
def load_global_fundamentals(include: Dict[str, bool] | None = None) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    all_keys = [
        # 市场结构/交易相关（全局）
        "premium","altcoin","reserve","correlation",
        # 链上/周期
        "ahr999","active_addresses","new_addresses","nupl",
        "sth_sopr","lth_sopr","sth_realized_price","lth_realized_price",
        "lth_supply","sth_supply","macro_oscillator","rhodl",
        # 宏观配套
        "dominance","m2_global","m2_us","stablecoin_mcap",
        # 周期类视觉指标
        "bubble_index","rainbow_chart","pi_cycle","golden_ratio",
        "puell_multiple","stock_flow",
        # ETF
        "etf_btc","etf_eth",
    ]

    if include is None:
        # 未提供 include -> 默认全部开启
        inc = {k: True for k in all_keys}
    else:
        # 提供 include -> 未指定的键一律 False（默认关闭）
        inc = {k: False for k in all_keys}
        for k, v in (include or {}).items():
            if k in inc:
                inc[k] = bool(v)

    # Coinbase 溢价
    if inc.get("premium", True):
        f = "data/cglass/index/premium-index.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = _rename_cols(df, {
                "premium": "premium_premium_index",
                "premium_rate": "premium_rate_premium_index",
            })
            df = df[["timestamp", "premium_premium_index", "premium_rate_premium_index"]]
            rows.append(df)

    # 山寨季节
    if inc.get("altcoin", True):
        f = "data/cglass/index/altcoin-season.csv"
        df = _read_csv_head(f)
        if not df.empty and {"timestamp", "altcoin_index", "altcoin_marketcap"}.issubset(df.columns):
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "altcoin_index": "altcoin_index_index",
                "altcoin_marketcap": "altcoin_marketcap_index",
            })
            df = df[["timestamp", "altcoin_index_index", "altcoin_marketcap_index"]]
            rows.append(df)

    # 比特币 Reserve Risk（作为全市场情绪代理）
    if inc.get("reserve", True):
        f = "data/cglass/index/bitcoin-reserve_risk.csv"
        df = _read_csv_head(f)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = _rename_cols(df, {
                "reserve_risk_index": "reserve_risk_index_bitcoin_bitcoin_reserve_risk",
                "movcd": "movcd_bitcoin_bitcoin_reserve_risk",
                "hodl_bank": "hodl_bank_bitcoin_bitcoin_reserve_risk",
                "vocd": "vocd_bitcoin_bitcoin_reserve_risk",
            })
            keep = [c for c in [
                "reserve_risk_index_bitcoin_bitcoin_reserve_risk",
                "movcd_bitcoin_bitcoin_reserve_risk",
                "hodl_bank_bitcoin_bitcoin_reserve_risk",
                "vocd_bitcoin_bitcoin_reserve_risk",
            ] if c in df.columns]
            df = df[["timestamp", *keep]]
            rows.append(df)

    # AHR999
    if inc.get("ahr999", True):
        f = "data/cglass/index/ahr999.csv"
        df = _read_csv_head(f)
        if not df.empty:
            if "timestamp" not in df.columns and "time" in df.columns:
                df["timestamp"] = df["time"]
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "average_price": "average_price_ahr999",
                "ahr999_value": "ahr999_value_ahr999",
                "current_value": "current_value_ahr999",
            })
            keep = [c for c in ["average_price_ahr999","ahr999_value_ahr999","current_value_ahr999"] if c in df.columns]
            df = df[["timestamp", *keep]]
            rows.append(df)

    # 活跃/新增地址
    if inc.get("active_addresses", True):
        f = "data/cglass/index/bitcoin-active_addresses.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"active_address_count": "active_address_count_bitcoin_active_addresses"})
            df = df[["timestamp","active_address_count_bitcoin_active_addresses"]]
            rows.append(df)
    if inc.get("new_addresses", True):
        f = "data/cglass/index/bitcoin-new_addresses.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"new_address_count": "new_address_count_bitcoin_new_addresses"})
            df = df[["timestamp","new_address_count_bitcoin_new_addresses"]]
            rows.append(df)

    # NUPL
    if inc.get("nupl", True):
        f = "data/cglass/index/bitcoin-net_unrealized_profit_loss.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"net_unpnl": "net_unpnl_bitcoin_bitcoin_net_unrealized_profit_loss"})
            df = df[["timestamp","net_unpnl_bitcoin_bitcoin_net_unrealized_profit_loss"]]
            rows.append(df)

    # SOPR & Realized Price & Holder Supply
    if inc.get("sth_sopr", True):
        f = "data/cglass/index/bitcoin-sth_sopr.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"sth_sopr": "sth_sopr_bitcoin_sth_sopr"})
            df = df[["timestamp","sth_sopr_bitcoin_sth_sopr"]]
            rows.append(df)
    if inc.get("lth_sopr", True):
        f = "data/cglass/index/bitcoin-lth_sopr.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"lth_sopr": "lth_sopr_bitcoin_lth_sopr"})
            df = df[["timestamp","lth_sopr_bitcoin_lth_sopr"]]
            rows.append(df)
    if inc.get("sth_realized_price", True):
        f = "data/cglass/index/bitcoin-sth_realized_price.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"sth_realized_price": "sth_realized_price_bitcoin_sth_realized_price"})
            df = df[["timestamp","sth_realized_price_bitcoin_sth_realized_price"]]
            rows.append(df)
    if inc.get("lth_realized_price", True):
        f = "data/cglass/index/bitcoin-lth_realized_price.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"lth_realized_price": "lth_realized_price_bitcoin_lth_realized_price"})
            df = df[["timestamp","lth_realized_price_bitcoin_lth_realized_price"]]
            rows.append(df)
    if inc.get("lth_supply", True):
        f = "data/cglass/index/bitcoin-long_term_holder_supply.csv"
        df = _read_csv_head(f)
        if not df.empty:
            # column order is value, timestamp; fix
            if "timestamp" not in df.columns:
                df.columns = ["long_term_holder_supply","timestamp"]
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"long_term_holder_supply": "long_term_holder_supply_bitcoin_bitcoin_long_term_holder_supply"})
            df = df[["timestamp","long_term_holder_supply_bitcoin_bitcoin_long_term_holder_supply"]]
            rows.append(df)
    if inc.get("sth_supply", True):
        f = "data/cglass/index/bitcoin-short_term_holder_supply.csv"
        df = _read_csv_head(f)
        if not df.empty:
            if "timestamp" not in df.columns:
                df.columns = ["short_term_holder_supply","timestamp"]
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"short_term_holder_supply": "short_term_holder_supply_bitcoin_bitcoin_short_term_holder_supply"})
            df = df[["timestamp","short_term_holder_supply_bitcoin_bitcoin_short_term_holder_supply"]]
            rows.append(df)

    # 宏观/市场配套
    if inc.get("macro_oscillator", True):
        f = "data/cglass/index/bitcoin-macro_oscillator.csv"
        df = _read_csv_head(f)
        if not df.empty:
            if "timestamp" not in df.columns:
                df.columns = ["bmo_value","timestamp"]
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"bmo_value": "bmo_value_bitcoin_bitcoin_macro_oscillator"})
            df = df[["timestamp","bmo_value_bitcoin_bitcoin_macro_oscillator"]]
            rows.append(df)
    if inc.get("rhodl", True):
        f = "data/cglass/index/bitcoin-rhodl_ratio.csv"
        df = _read_csv_head(f)
        if not df.empty:
            if "timestamp" not in df.columns:
                df.columns = ["rhodl_ratio","timestamp"]
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"rhodl_ratio": "rhodl_ratio_bitcoin_bitcoin_rhodl_ratio"})
            df = df[["timestamp","rhodl_ratio_bitcoin_bitcoin_rhodl_ratio"]]
            rows.append(df)
    if inc.get("dominance", True):
        f = "data/cglass/index/bitcoin-dominance.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "market_cap": "market_cap_bitcoin_bitcoin_dominance",
                "bitcoin_dominance": "bitcoin_dominance_bitcoin_bitcoin_dominance",
            })
            keep = ["market_cap_bitcoin_bitcoin_dominance","bitcoin_dominance_bitcoin_bitcoin_dominance"]
            df = df[["timestamp", *[c for c in keep if c in df.columns]]]
            rows.append(df)
    if inc.get("m2_global", True):
        f = "data/cglass/index/bitcoin-vs-global-m2-growth.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "global_m2_yoy_growth": "global_m2_yoy_growth_bitcoin_bitcoin_vs_global_m2_growth",
                "global_m2_supply": "global_m2_supply_bitcoin_bitcoin_vs_global_m2_growth",
            })
            keep = ["global_m2_yoy_growth_bitcoin_bitcoin_vs_global_m2_growth","global_m2_supply_bitcoin_bitcoin_vs_global_m2_growth"]
            df = df[["timestamp", *[c for c in keep if c in df.columns]]]
            rows.append(df)
    if inc.get("m2_us", True):
        f = "data/cglass/index/bitcoin-vs-us-m2-growth.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "us_m2_yoy_growth": "us_m2_yoy_growth_bitcoin_bitcoin_vs_us_m2_growth",
                "us_m2_supply": "us_m2_supply_bitcoin_bitcoin_vs_us_m2_growth",
            })
            keep = ["us_m2_yoy_growth_bitcoin_bitcoin_vs_us_m2_growth","us_m2_supply_bitcoin_bitcoin_vs_us_m2_growth"]
            df = df[["timestamp", *[c for c in keep if c in df.columns]]]
            rows.append(df)
    if inc.get("stablecoin_mcap", True):
        f = "data/cglass/index/stableCoin-marketCap-history.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"total": "stablecoin_marketcap_total"})
            df = df[["timestamp","stablecoin_marketcap_total"]]
            rows.append(df)

    # 周期类视觉指标
    if inc.get("bubble_index", True):
        f = "data/cglass/index/bitcoin/bubble_index.csv"
        df = _read_csv_head(f)
        if not df.empty:
            if "timestamp" not in df.columns:
                # reorder
                cols = list(df.columns)
                if "date_string" in cols and "timestamp" not in cols:
                    # try
                    pass
            df["timestamp"] = _ensure_int_ms(df.get("timestamp")).astype("int64")
            df = df.rename(columns={
                "bubble_index": "bubble_index_bitcoin_bubble_index",
                "google_trend_percent": "google_trend_percent_bitcoin_bubble_index",
                "mining_difficulty": "mining_difficulty_bitcoin_bubble_index",
                "transaction_count": "transaction_count_bitcoin_bubble_index",
                "address_send_count": "address_send_count_bitcoin_bubble_index",
                "tweet_count": "tweet_count_bitcoin_bubble_index",
            })
            keep = [c for c in [
                "bubble_index_bitcoin_bubble_index",
                "google_trend_percent_bitcoin_bubble_index",
                "mining_difficulty_bitcoin_bubble_index",
                "transaction_count_bitcoin_bubble_index",
                "address_send_count_bitcoin_bubble_index",
                "tweet_count_bitcoin_bubble_index",
            ] if c in df.columns]
            df = df[["timestamp", *keep]]
            rows.append(df)
    if inc.get("rainbow_chart", True):
        f = "data/cglass/index/bitcoin/rainbow_chart.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            # 直接保留各色带，统一加后缀
            rename = {k: f"{k}_bitcoin_rainbow_chart" for k in df.columns if k != "timestamp"}
            df = df.rename(columns=rename)
            keep = [c for c in df.columns if c != "timestamp"]
            df = df[["timestamp", *keep]]
            rows.append(df)
    if inc.get("pi_cycle", True):
        f = "data/cglass/index/pi_cycle_indicator.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "ma_110": "ma_110_pi_cycle_indicator",
                "ma_350_mu_2": "ma_350_mu_2_pi_cycle_indicator",
            })
            keep = [c for c in ["ma_110_pi_cycle_indicator","ma_350_mu_2_pi_cycle_indicator"] if c in df.columns]
            df = df[["timestamp", *keep]]
            rows.append(df)
    if inc.get("golden_ratio", True):
        f = "data/cglass/index/golden_ratio_multiplier.csv"
        df = _read_csv_head(f)
        if not df.empty:
            if "timestamp" not in df.columns:
                # file order observed: low_bull_high_2,timestamp,ma_350,accumulation_high_1_6,x_3,x_5,x_8,x_13,x_21
                pass
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "ma_350": "ma_350_golden_ratio_multiplier",
                "low_bull_high_2": "low_bull_high_2_golden_ratio_multiplier",
                "accumulation_high_1_6": "accumulation_high_1_6_golden_ratio_multiplier",
                "x_3": "x_3_golden_ratio_multiplier",
                "x_5": "x_5_golden_ratio_multiplier",
                "x_8": "x_8_golden_ratio_multiplier",
                "x_13": "x_13_golden_ratio_multiplier",
                "x_21": "x_21_golden_ratio_multiplier",
            })
            keep = [c for c in [
                "ma_350_golden_ratio_multiplier",
                "low_bull_high_2_golden_ratio_multiplier",
                "accumulation_high_1_6_golden_ratio_multiplier",
                "x_3_golden_ratio_multiplier","x_5_golden_ratio_multiplier","x_8_golden_ratio_multiplier",
                "x_13_golden_ratio_multiplier","x_21_golden_ratio_multiplier",
            ] if c in df.columns]
            df = df[["timestamp", *keep]]
            rows.append(df)
    if inc.get("puell_multiple", True):
        f = "data/cglass/index/puell_multiple.csv"
        df = _read_csv_head(f)
        if not df.empty:
            if "timestamp" not in df.columns:
                df.columns = ["timestamp","puell_multiple"]
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"puell_multiple": "puell_multiple_puell_multiple"})
            df = df[["timestamp","puell_multiple_puell_multiple"]]
            rows.append(df)
    if inc.get("stock_flow", True):
        f = "data/cglass/index/stock_flow.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={"next_halving": "next_halving_stock_flow"})
            df = df[["timestamp","next_halving_stock_flow"]]
            rows.append(df)

    # 跨市场相关性（可选，不默认）
    if inc.get("correlation", False):
        f = "data/cglass/index/bitcoin-correlation.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            ren = {c: f"correlation_{c}" for c in df.columns if c != "timestamp"}
            df = df.rename(columns=ren)
            keep = [c for c in df.columns if c != "timestamp"]
            df = df[["timestamp", *keep]]
            rows.append(df)

    # ETF（全局）
    if inc.get("etf_btc", True):
        f = "data/cglass/futures/futures-etf-btc.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "net_assets_usd": "etf_btc_net_assets_usd",
                "change_usd": "etf_btc_change_usd",
                "price_usd": "etf_btc_price_usd",
            })
            keep = [c for c in ["etf_btc_net_assets_usd","etf_btc_change_usd","etf_btc_price_usd"] if c in df.columns]
            df = df[["timestamp", *keep]]
            rows.append(df)
    if inc.get("etf_eth", True):
        f = "data/cglass/futures/futures-etf-eth.csv"
        df = _read_csv_head(f)
        if not df.empty:
            df["timestamp"] = _ensure_int_ms(df["timestamp"]).astype("int64")
            df = df.rename(columns={
                "net_assets_usd": "etf_eth_net_assets_usd",
                "change_usd": "etf_eth_change_usd",
            })
            keep = [c for c in ["etf_eth_net_assets_usd","etf_eth_change_usd"] if c in df.columns]
            df = df[["timestamp", *keep]]
            rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["timestamp"])  # 空表
    out = rows[0]
    for d in rows[1:]:
        out = out.merge(d, on=["timestamp"], how="outer")
    out = out.sort_values(["timestamp"]).reset_index(drop=True)
    return out


# --------------- 主融合 ---------------
def fuse(
    base_csv: str = "data/merged/full_merged.csv",
    out_csv: str = "data/merged/full_merged_with_fundamentals.csv",
    slim_csv: str | None = "data/merged/full_merged_slim.csv",
    cols_txt: str | None = "data/merged/fundamental_columns.txt",
    *,
    include_symbol: Dict[str, bool] | None = None,
    include_global: Dict[str, bool] | None = None,
    validate: bool = True,
    report_dir: str | None = "data/merged/fuse_audit",
    extra_field: Dict[str, Any] | None = None,
    no_nan_policy: Dict[str, Any] | None = None,
    dataset_type: str | None = None,
    max_missing_ratio: float | None = None,
) -> Tuple[str, str | None, str | None]:
    """Fuse base price/technical data with fundamentals and optional on-chain series.

    Parameters
    - base_csv: merged technical/base CSV to start from
    - out_csv/slim_csv/cols_txt: output paths (final/simplified/new-columns)
    - include_symbol: per-symbol (derivatives/spot) feature switches
    - include_global: on-chain/macro feature switches (per expert)
    - validate/report_dir: output coverage/time-audit reports when True
    - extra_field: naming/grouping config for outputs (folder/name suffix)
    - dataset_type: 'fundamentals_only' | 'combined'
    - no_nan_policy: {enabled, scope('onchain'|'all_new'|'custom'),
                      method('intersect'|'drop'), columns: []}
    - max_missing_ratio: drop newly added columns whose missing ratio exceeds this threshold (0-1)
      - intersect: for each (symbol,period), crops to the longest
        continuous window where target columns are all non-null
      - drop: simply removes rows with any nulls in target columns
    Returns: (out_csv, slim_csv, cols_txt)
    """
    assert os.path.exists(base_csv), f"base not found: {base_csv}"

    print(f"↻ load base: {base_csv}")
    base = pd.read_csv(base_csv, low_memory=False)
    base["timestamp"] = _ensure_int_ms(base["timestamp"]).astype("int64")

    # 确定符号集合
    symbols = sorted(base["symbol"].dropna().unique().tolist())
    print(f"symbols: {len(symbols)} -> {symbols[:5]}{'...' if len(symbols)>5 else ''}")

    # 拼接各币种 1d 基本面
    per_symbol_rows = []
    for s in symbols:
        df = load_symbol_fundamentals(s, include=include_symbol)
        if not df.empty:
            per_symbol_rows.append(df)
    sym_fund = pd.concat(per_symbol_rows, ignore_index=True) if per_symbol_rows else pd.DataFrame(columns=["timestamp","symbol"])

    # 根据数据集类型控制是否纳入链上（全局）指标
    if dataset_type and dataset_type.lower() in ("fundamentals_only","fund_only","funds_only"):
        glb = pd.DataFrame(columns=["timestamp"])  # 禁用链上合入
    else:
        glb = load_global_fundamentals(include=include_global)

    # 左连接 + 按 symbol 前向填充（广播）
    print("↻ merge fundamentals into base ...")
    merged = base
    base_cols = set(base.columns)
    if not sym_fund.empty:
        merged = merged.merge(sym_fund, on=["timestamp", "symbol"], how="left")
    after_fund_cols = set(merged.columns)
    if not glb.empty:
        merged = merged.merge(glb, on=["timestamp"], how="left")
    after_all_cols = set(merged.columns)

    fund_cols = list(after_fund_cols - base_cols)
    onchain_cols = list(after_all_cols - after_fund_cols)

    # 额外：把标签写入文件名后缀（不进 CSV）
    def _compose_suffix(cfg: Dict[str, Any] | None) -> str:
        if not cfg or not isinstance(cfg, dict):
            return ""
        name = str(cfg.get('name') or '').strip()
        value = cfg.get('value')
        how = str(cfg.get('append_to_filename', 'name')).strip().lower()  # none|name|value|both
        if not name:
            return ""
        if how in ("none", "off", "no", "false"):
            return ""
        if how == 'both' and value is not None:
            return f"_{name}-{value}"
        if how == 'value' and value is not None:
            return f"_{value}"
        # 默认仅 name
        return f"_{name}"

    def _with_suffix(path: str | None, suffix: str) -> str | None:
        if not path or not suffix:
            return path
        d = os.path.dirname(path)
        b = os.path.basename(path)
        stem, ext = os.path.splitext(b)
        return os.path.join(d, f"{stem}{suffix}{ext}")

    def _compose_tag_for_folder(cfg: Dict[str, Any] | None) -> str:
        if not cfg or not isinstance(cfg, dict):
            return ""
        name = str(cfg.get('name') or '').strip()
        value = cfg.get('value')
        mode = str(cfg.get('folder_mode', cfg.get('append_to_filename', 'name'))).strip().lower()
        if not name:
            return ""
        if mode == 'both' and value is not None:
            return f"{name}-{value}"
        if mode == 'value' and value is not None:
            return f"{value}"
        return f"{name}"

    def _into_group_folder(path: str | None, cfg: Dict[str, Any] | None) -> str | None:
        if not path:
            return path
        if not cfg or not cfg.get('group_to_folder'):
            return path
        tag = _compose_tag_for_folder(cfg)
        if not tag:
            return path
        base_root = cfg.get('folder_root')
        if base_root:
            # 始终在指定根目录下再按 tag 建子目录
            root = os.path.join(base_root, tag)
            os.makedirs(root, exist_ok=True)
            return os.path.join(root, os.path.basename(path))
        d = os.path.dirname(path)
        gdir = os.path.join(d, tag)
        os.makedirs(gdir, exist_ok=True)
        return os.path.join(gdir, os.path.basename(path))

    # 需要前向填充的列集合（新加列）
    new_cols = [c for c in merged.columns if c not in base.columns]
    if new_cols:
        merged = merged.sort_values(["symbol", "timestamp"])  # 保障时序
        merged[new_cols] = (
            merged.groupby("symbol", sort=False)[new_cols]
                  .apply(lambda df: df.ffill())
                  .reset_index(level=0, drop=True)
        )

    missing_threshold_cols: List[str] = []
    missing_threshold_rows: int = 0
    threshold: float | None = None
    if max_missing_ratio is not None:
        try:
            threshold = float(max_missing_ratio)
        except (TypeError, ValueError):
            threshold = None
        else:
            if threshold < 0:
                threshold = None
    if threshold is not None and new_cols:
        missing_ratio = merged[new_cols].isna().mean()
        focus_cols = missing_ratio[missing_ratio > threshold].index.tolist()
        if focus_cols:
            mask = merged[focus_cols].isna().any(axis=1)
            missing_threshold_rows = int(mask.sum())
            if missing_threshold_rows > 0:
                merged = merged.loc[~mask].reset_index(drop=True)
            missing_threshold_cols = focus_cols
            preview = focus_cols[:5]
            suffix = '...' if len(focus_cols) > 5 else ''
            print(f"[warn] dropped {missing_threshold_rows} row(s) due to columns exceeding missing ratio {threshold:.2%}: {preview}{suffix}")
    else:
        missing_threshold_cols = []
        missing_threshold_rows = 0

    # 可选：对选定列执行“无缺失”裁剪（按组取交集窗口或直接丢行）
    target_cols_used: List[str] = []
    all_null_target_cols: List[str] = []
    if no_nan_policy and (no_nan_policy.get('enabled') is True):
        scope = str(no_nan_policy.get('scope','onchain')).lower()  # onchain|all_new|custom
        method = str(no_nan_policy.get('method','intersect')).lower()  # intersect|drop
        custom_cols = no_nan_policy.get('columns') or []
        if scope == 'onchain':
            target_cols = [c for c in onchain_cols if c in merged.columns]
        elif scope == 'all_new':
            target_cols = [c for c in new_cols if c in merged.columns]
        else:
            target_cols = [c for c in custom_cols if c in merged.columns]

        if target_cols:
            non_null_cols = [c for c in target_cols if merged[c].notna().any()]
            all_null_target_cols = [c for c in target_cols if c not in non_null_cols]
            if all_null_target_cols:
                preview = all_null_target_cols[:5]
                preview_suffix = '...' if len(all_null_target_cols) > 5 else ''
                print(f"[warn] no_nan_policy skipped {len(all_null_target_cols)} all-null column(s): {preview}{preview_suffix}")
            target_cols = non_null_cols

        target_cols_used = list(target_cols)

        if target_cols:
            merged = merged.sort_values(["symbol","period","timestamp"]).reset_index(drop=True)
            groups = []
            for (sym, per), g in merged.groupby(["symbol","period"], sort=False):
                g = g.copy()
                if method == 'intersect':
                    valid = g[target_cols].notna().all(axis=1).values
                    if valid.any():
                        import numpy as np
                        first = int(np.argmax(valid))
                        last = int(len(valid) - 1 - np.argmax(valid[::-1]))
                        sub = g.iloc[first:last+1]
                        sub = sub[sub[target_cols].notna().all(axis=1)]
                        groups.append(sub)
                    # else: drop entire group (no coverage)
                else:  # drop
                    groups.append(g[g[target_cols].notna().all(axis=1)])
            merged = pd.concat(groups, ignore_index=True) if groups else merged.iloc[0:0]
        elif all_null_target_cols:
            print('[warn] no_nan_policy: all candidate columns were empty; skip intersection to keep base rows.')

    # 最后阶段：根据 extra_field 重写输出文件名
    suffix = _compose_suffix(extra_field)
    out_csv = _with_suffix(out_csv, suffix)
    slim_csv = _with_suffix(slim_csv, suffix) if slim_csv else None
    cols_txt = _with_suffix(cols_txt, suffix) if cols_txt else None
    report_dir = _with_suffix(report_dir, suffix) if report_dir else None

    # 可选：将所有输出放入分组子目录
    out_csv = _into_group_folder(out_csv, extra_field)
    slim_csv = _into_group_folder(slim_csv, extra_field) if slim_csv else None
    cols_txt = _into_group_folder(cols_txt, extra_field) if cols_txt else None
    report_dir = _into_group_folder(report_dir, extra_field) if report_dir else None

    # 保存 enriched
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"✅ enriched saved -> {out_csv}")

    # 导出列清单（仅新增列；含基本面与链上）
    if cols_txt and new_cols:
        with open(cols_txt, "w", encoding="utf-8") as f:
            for c in new_cols:
                f.write(c + "\n")
        print(f"📝 columns list -> {cols_txt}")

    # 精简列版本（便于分享）：时间/标识 + 关键价量 + 基本面 + 目标
    if slim_csv:
        core_id = ["timestamp","datetime","symbol","period","time_idx"]
        core_px = [c for c in ["open","high","low","close","volume"] if c in merged.columns]
        core_target = [c for c in [
            "target_logreturn","target_binarytrend","target_logsharpe_ratio",
            "logreturn_pct_rank","target_return_outlier_lower_10","target_return_outlier_upper_90",
            "target_max_drawdown","target_drawdown_prob","target_trend_persistence",
            "target_pullback_prob","target_sideway_detect","target_breakout_count",
        ] if c in merged.columns]
        slim_cols = [*core_id, *core_px]
        slim_cols += new_cols  # 全部基本面
        slim_cols += core_target
        slim_cols = [c for c in slim_cols if c in merged.columns]
        merged[slim_cols].to_csv(slim_csv, index=False)
        print(f"📦 slim saved -> {slim_csv} (cols={len(slim_cols)})")

    # 每专家的交集起点/样本量汇总
    try:
        out_dir = os.path.dirname(out_csv)
        grp = merged.groupby(["symbol","period"], dropna=False)
        summary = grp.agg(
            rows=("timestamp","count"),
            start_ts=("timestamp","min"),
            end_ts=("timestamp","max")
        ).reset_index()
        summary["start_dt"] = pd.to_datetime(summary["start_ts"], unit="ms", utc=True)
        summary["end_dt"] = pd.to_datetime(summary["end_ts"], unit="ms", utc=True)
        # 记录策略与列数
        policy = no_nan_policy or {}
        summary["policy_enabled"] = bool(policy.get("enabled", False))
        summary["policy_scope"] = str(policy.get("scope", "")).strip()
        summary["policy_method"] = str(policy.get("method", "")).strip()
        summary["intersect_cols_count"] = len(target_cols_used)
        summary["intersect_all_null_cols_count"] = len(all_null_target_cols)
        summary["missing_threshold_cols_count"] = len(missing_threshold_cols)
        summary["missing_threshold_rows"] = missing_threshold_rows
        summ_path = os.path.join(out_dir, "dataset_group_summary.csv")
        summary[["symbol","period","rows","start_dt","end_dt","policy_enabled","policy_scope","policy_method","intersect_cols_count","intersect_all_null_cols_count","missing_threshold_cols_count","missing_threshold_rows"]] \
            .to_csv(summ_path, index=False)
        if target_cols_used:
            with open(os.path.join(out_dir, "intersect_columns.txt"), "w", encoding="utf-8") as fh:
                for c in target_cols_used:
                    fh.write(c+"\n")
        if all_null_target_cols:
            with open(os.path.join(out_dir, "intersect_columns_all_null.txt"), "w", encoding="utf-8") as fh:
                for c in all_null_target_cols:
                    fh.write(c+"\n")
        if missing_threshold_cols:
            with open(os.path.join(out_dir, "missing_threshold_columns.txt"), "w", encoding="utf-8") as fh:
                for c in missing_threshold_cols:
                    fh.write(c+"\n")
        print(f"🧾 group summary -> {summ_path}")
    except Exception as e:
        print(f"[warn] write group summary failed: {e}")

    # 校验报告
    if validate and new_cols:
        rep = report_dir or os.path.join(os.path.dirname(out_csv), "fuse_audit")
        os.makedirs(rep, exist_ok=True)

        # 覆盖率（总体）
        cov = (
            merged[new_cols].notna().agg(['count']).T
            .rename(columns={'count': 'non_null'})
        )
        cov['total_rows'] = len(merged)
        cov['non_null_ratio'] = cov['non_null'] / cov['total_rows']
        cov.insert(0, 'column', cov.index)
        cov = cov.sort_values('non_null_ratio')
        cov.to_csv(os.path.join(rep, 'coverage_overall.csv'), index=False)

        # 覆盖率（按 period）
        if 'period' in merged.columns:
            per_rows = []
            for p, g in merged.groupby('period'):
                r = g[new_cols].notna().mean().rename('ratio').to_frame()
                r.insert(0, 'column', r.index)
                r['period'] = p
                per_rows.append(r.reset_index(drop=True))
            if per_rows:
                per = pd.concat(per_rows, ignore_index=True)
                per.to_csv(os.path.join(rep, 'coverage_by_period.csv'), index=False)

        # 覆盖率（按 symbol+period）
        sym_rows = []
        if {'symbol','period'}.issubset(merged.columns):
            for (s, p), g in merged.groupby(['symbol','period']):
                r = g[new_cols].notna().mean().rename('ratio').to_frame()
                r.insert(0, 'column', r.index)
                r['symbol'] = s
                r['period'] = p
                sym_rows.append(r.reset_index(drop=True))
        if sym_rows:
            sp = pd.concat(sym_rows, ignore_index=True)
            sp.to_csv(os.path.join(rep, 'coverage_by_symbol_period.csv'), index=False)

        print(f"🧪 validation reports -> {rep}")

        # 可选：时间连续性审计（依赖本地工具，不存在则跳过）
        try:
            from src.prune_and_time_audit import audit_time
            sum_df, gaps_df = audit_time(merged, out_csv)
            if not sum_df.empty:
                sum_df.to_csv(os.path.join(rep, 'time_audit_summary.csv'), index=False)
            if not gaps_df.empty:
                gaps_df.to_csv(os.path.join(rep, 'time_audit_gaps_first2000.csv'), index=False)
            print(f"🧭 time audit -> {rep}")
        except Exception as e:
            print(f"[warn] time audit skipped: {e}")

    return out_csv, slim_csv, cols_txt


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    if 'yaml' in globals() and yaml is not None:
        return yaml.safe_load(text) or {}
    # 简易降级：实现一个极简 YAML 解析（支持本文件结构：key: value / 嵌套 dict，不支持列表、复杂类型）
    def _parse_value(v: str) -> Any:
        vs = v.strip()
        if vs.lower() in ("true", "yes", "on"): return True
        if vs.lower() in ("false", "no", "off"): return False
        if vs.lower() in ("null", "none", "~"): return None
        if (vs.startswith('"') and vs.endswith('"')) or (vs.startswith("'") and vs.endswith("'")):
            return vs[1:-1]
        try:
            if vs.isdigit() or (vs.startswith('-') and vs[1:].isdigit()):
                return int(vs)
            fv = float(vs)
            return fv
        except Exception:
            return vs

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip():
            continue
        # 去掉注释
        line = raw.split('#', 1)[0]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        line = line.strip()
        if ':' not in line:
            continue
        key, rest = line.split(':', 1)
        key = key.strip()
        rest = rest.strip()
        # 回退到当前缩进层
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if rest == '':
            # 新的子 dict 层
            d: Dict[str, Any] = {}
            parent[key] = d
            stack.append((indent, d))
        else:
            parent[key] = _parse_value(rest)
    return root


def _dump_config_snapshot(dir_path: str, data: Dict[str, Any], fname: str = 'config_snapshot.yaml') -> None:
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, fname)
    # 优先 YAML，降级 JSON
    try:
        import yaml  # type: ignore
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"📝 snapshot -> {path}")
        return
    except Exception:
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"📝 snapshot(json) -> {path}")


def _coerce_periods(x: Any) -> List[str] | None:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        s = x.strip().strip('[]')
        if not s:
            return None
        return [t.strip().strip("'\"") for t in s.split(',') if t.strip()]
    return None


def _post_convert_full(out_csv: str, post_cfg: Dict[str, Any] | None) -> None:
    if not post_cfg:
        return
    try:
        import pandas as pd
        from pathlib import Path
        periods = _coerce_periods(post_cfg.get('periods')) or ["1h","4h","1d"]
        src = Path(out_csv)
        if not src.exists():
            return
        df = pd.read_csv(src)
        if periods and 'period' in df.columns:
            df = df[df['period'].isin(periods)]
        if post_cfg.get('pkl'):
            pkl_path = src.with_suffix('.pkl')
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(pkl_path)
            print(f"✅ post-convert pkl -> {pkl_path}")
        if post_cfg.get('parquet'):
            try:
                pq_path = src.with_suffix('.parquet')
                pq_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(pq_path, index=False)
                print(f"✅ post-convert parquet -> {pq_path}")
            except Exception as e:
                print(f"[warn] parquet convert failed: {e}")
    except Exception as e:
        print(f"[warn] post_convert failed: {e}")

def _aggregate_expert_summaries(results: List[Tuple[str, str | None, str | None]], cfg: Dict[str, Any]) -> None:
    try:
        import pandas as pd
        rows = []
        for out_csv, _, _ in results:
            d = os.path.dirname(out_csv)
            expert = os.path.basename(d)
            summ = os.path.join(d, 'dataset_group_summary.csv')
            if os.path.exists(summ):
                df = pd.read_csv(summ)
                df.insert(0, 'expert', expert)
                rows.append(df)
        if not rows:
            return
        all_df = pd.concat(rows, ignore_index=True)
        rep_root = os.path.join('reports', 'datasets')
        os.makedirs(rep_root, exist_ok=True)
        out_path = os.path.join(rep_root, 'experts_group_summary.csv')
        all_df.to_csv(out_path, index=False)
        print(f"📚 aggregated experts summary -> {out_path}")
    except Exception as e:
        print(f"[warn] aggregate experts summary failed: {e}")


def run_with_config(cfg_path: str = 'pipelines/configs/fuse_fundamentals.yaml') -> Tuple[str, str | None, str | None] | List[Tuple[str, str | None, str | None]]:
    if not os.path.exists(cfg_path):
        print(f"[warn] config not found: {cfg_path}; use defaults")
        return fuse()
    cfg = _load_yaml(cfg_path)

    experts = cfg.get('experts')
    if experts and isinstance(experts, list) and len(experts) > 0:
        results: List[Tuple[str, str | None, str | None]] = []
        for ex in experts:
            name = str(ex.get('name') or '').strip()
            if not name:
                continue
            base_csv = cfg.get('base_csv', 'data/merged/full_merged.csv')
            out_csv = cfg.get('out_csv', 'data/merged/full_merged_with_fundamentals.csv')
            slim_csv = cfg.get('slim_csv', 'data/merged/full_merged_slim.csv')
            cols_txt = cfg.get('cols_txt', 'data/merged/fundamental_columns.txt')
            validate = bool(cfg.get('validate', True))
            report_dir = cfg.get('report_dir', 'data/merged/fuse_audit')

            # merge include overrides
            inc_sym = dict(cfg.get('include_symbol') or {})
            inc_sym.update(ex.get('include_symbol') or {})
            inc_glb = dict(cfg.get('include_global') or {})
            inc_glb_src = ex.get('include_global')
            if isinstance(inc_glb_src, dict):
                inc_glb.update(inc_glb_src)

            # dataset type & no-nan policy
            ds_type = ex.get('dataset_type', cfg.get('dataset_type'))
            nn_policy = ex.get('no_nan_policy', cfg.get('no_nan_policy'))

            # extra field for folder naming
            ef = dict(cfg.get('extra_field') or {})
            ef['name'] = name

            out_csv_i, slim_csv_i, cols_txt_i = fuse(
                base_csv=base_csv,
                out_csv=out_csv,
                slim_csv=slim_csv,
                cols_txt=cols_txt,
                include_symbol=inc_sym,
                include_global=inc_glb,
                validate=validate,
                report_dir=report_dir,
                extra_field=ef,
                no_nan_policy=nn_policy,
                dataset_type=ds_type,
                max_missing_ratio=ex.get('max_missing_ratio', cfg.get('max_missing_ratio')),
            )

            # snapshot for this expert
            eff_cfg = {
                'name': name,
                'include_symbol': inc_sym,
                'include_global': inc_glb,
                'post_convert': ex.get('post_convert', cfg.get('post_convert')),
            }
            _dump_config_snapshot(os.path.dirname(out_csv_i), eff_cfg)

            # post convert for this expert (per-expert override > global)
            post_cfg = ex.get('post_convert', cfg.get('post_convert'))
            _post_convert_full(out_csv_i, post_cfg)

            results.append((out_csv_i, slim_csv_i, cols_txt_i))
        _aggregate_expert_summaries(results, cfg)
        return results

    # 支持字典形式：experts_map: { expert_name: { include_symbol:..., include_global:..., post_convert:... } }
    experts_map = cfg.get('experts_map')
    if experts_map and isinstance(experts_map, dict) and len(experts_map) > 0:
        results: List[Tuple[str, str | None, str | None]] = []
        selected = _coerce_periods(cfg.get('experts_selected'))  # reuse list parser for convenience
        keys = list(experts_map.keys())
        if selected:
            keys = [k for k in keys if k in selected]
        for name in keys:
            entry = experts_map.get(name)
            name = str(name).strip()
            if not name:
                continue

            cfg_path = None
            ex: Dict[str, Any]
            if isinstance(entry, str):
                cfg_path = entry
                ex = _load_yaml(entry) or {}
            elif isinstance(entry, dict):
                ex = dict(entry)
                cfg_path = ex.pop('config_path', None)
                if cfg_path:
                    file_cfg = _load_yaml(cfg_path) or {}
                    merged: Dict[str, Any] = dict(file_cfg)
                    merged.update(ex)
                    ex = merged
            else:
                ex = {}

            base_csv = cfg.get('base_csv', 'data/merged/full_merged.csv')
            out_csv = cfg.get('out_csv', 'data/merged/full_merged_with_fundamentals.csv')
            slim_csv = cfg.get('slim_csv', 'data/merged/full_merged_slim.csv')
            cols_txt = cfg.get('cols_txt', 'data/merged/fundamental_columns.txt')
            validate = bool(cfg.get('validate', True))
            report_dir = cfg.get('report_dir', 'data/merged/fuse_audit')

            inc_sym = dict(cfg.get('include_symbol') or {})
            inc_sym.update(ex.get('include_symbol') or {})
            inc_glb = dict(cfg.get('include_global') or {})
            inc_glb_src = ex.get('include_global')
            if isinstance(inc_glb_src, dict):
                inc_glb.update(inc_glb_src)

            ds_type = ex.get('dataset_type', cfg.get('dataset_type'))
            nn_policy = ex.get('no_nan_policy', cfg.get('no_nan_policy'))

            if ex.get('extra_field'):
                ef = dict(ex.get('extra_field') or {})
            else:
                ef = dict(cfg.get('extra_field') or {})
                ef.setdefault('name', name)

            out_csv_i, slim_csv_i, cols_txt_i = fuse(
                base_csv=base_csv,
                out_csv=out_csv,
                slim_csv=slim_csv,
                cols_txt=cols_txt,
                include_symbol=inc_sym,
                include_global=inc_glb,
                validate=validate,
                report_dir=report_dir,
                extra_field=ef,
                no_nan_policy=nn_policy,
                dataset_type=ds_type,
                max_missing_ratio=ex.get('max_missing_ratio', cfg.get('max_missing_ratio')),
            )
            eff_cfg = {
                'name': name,
                'include_symbol': inc_sym,
                'include_global': inc_glb,
                'post_convert': ex.get('post_convert', cfg.get('post_convert')),
            }
            if cfg_path:
                eff_cfg['config_path'] = cfg_path
            _dump_config_snapshot(os.path.dirname(out_csv_i), eff_cfg)
            _post_convert_full(out_csv_i, ex.get('post_convert', cfg.get('post_convert')))
            results.append((out_csv_i, slim_csv_i, cols_txt_i))
        _aggregate_expert_summaries(results, cfg)
        return results

    # single expert (legacy mode)
    base_csv = cfg.get('base_csv', 'data/merged/full_merged.csv')
    out_csv = cfg.get('out_csv', 'data/merged/full_merged_with_fundamentals.csv')
    slim_csv = cfg.get('slim_csv', 'data/merged/full_merged_slim.csv')
    cols_txt = cfg.get('cols_txt', 'data/merged/fundamental_columns.txt')
    validate = bool(cfg.get('validate', True))
    report_dir = cfg.get('report_dir', 'data/merged/fuse_audit')
    include_symbol = cfg.get('include_symbol') or None
    include_global = cfg.get('include_global') or None
    extra_field = cfg.get('extra_field') or None

    out_csv, slim_csv, cols_txt = fuse(
        base_csv=base_csv,
        out_csv=out_csv,
        slim_csv=slim_csv,
        cols_txt=cols_txt,
        include_symbol=include_symbol,
        include_global=include_global,
        validate=validate,
        report_dir=report_dir,
        extra_field=extra_field,
        max_missing_ratio=cfg.get('max_missing_ratio'),
    )
    _dump_config_snapshot(os.path.dirname(out_csv), {
        'name': (extra_field or {}).get('name'),
        'include_symbol': include_symbol,
        'include_global': include_global,
        'post_convert': cfg.get('post_convert'),
    })
    _post_convert_full(out_csv, cfg.get('post_convert'))
    return out_csv, slim_csv, cols_txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Fuse fundamentals into merged dataset (configurable via YAML).')
    ap.add_argument('--config', default='pipelines/configs/fuse_fundamentals.yaml', help='YAML config path')
    args = ap.parse_args()
    print(run_with_config(args.config))
