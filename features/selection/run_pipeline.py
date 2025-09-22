from __future__ import annotations
import argparse
import os
from .tree_perm import run as run_tree_perm
from .aggregate_core import aggregate_tree_perm, build_unified_core, export_selected_features, export_core_and_boost
from .filter_stage import run_filter_for_channel, FilterParams
from .embedded_stage import run_embedded_for_channel
from .common import load_split
import os
import pandas as pd


def _experts_default_field_map() -> dict:
    """基于 experts.md 推荐，给出每位专家的默认字段（通用名）。
    这里只列出核心集合；后续会做同义解析并映射到实际列名。
    """
    base_ohlcv = ["open", "high", "low", "close", "volume", "vwap"]
    alpha_dir_base = base_ohlcv + [
        "rsi", "macd_hist", "adx", "adx_scaled", "boll_bandwidth",
        "ma5", "ma20", "obv", "atr", "cci", "momentum", "ppo",
        "stoch_rsi", "williams_r", "tsi", "donchian_high_20", "donchian_low_20",
        "keltner_upper_20", "keltner_lower_20", "squeeze_on", "trendline_slope_20",
        "ma_diff_5_20", "boll_pctb", "amplitude_range",
    ]
    alpha_ret_base = alpha_dir_base + [
        "rolling_volatility", "parkinson_volatility", "ewmavolatility", "return_skewness",
        "atr_ratio_price", "atr_slope",
    ]
    risk_base = base_ohlcv + [
        "var_95_30d", "cvar_95_30d", "ret_skew_30d", "ret_kurt_30d", "atr", "atr_change", "atr_ratio_range",
    ]
    regime_base = base_ohlcv + [
        "trend_flag", "range_flag", "high_vol_flag", "stress_flag",
        "volume_profile_hvn", "volume_profile_lvn", "volume_relative_20", "volume_percentile",
    ]
    structure_base = base_ohlcv + [
        "support_level_20", "resistance_level_20", "breakout_recent_high", "pullback_from_high",
        "distance_to_support", "distance_to_resistance", "volume_spike_near_support", "volume_spike_near_resist",
        "tail_bias_norm", "amplitude_range",
    ]
    relative_base = base_ohlcv + [
        "r_sym", "r_btc", "beta_btc_60d", "price_position_ma200",
    ]
    # rich 通用占位（若缺失将记录到报表）
    rich_common = [
        "funding_rate", "oi_total", "taker_buy_sell_imbalance", "basis_z", "etf_flow", "etf_premium",
        "active_addresses", "new_addresses", "stablecoin_mcap", "exch_netflow", "cb_premium",
    ]
    return {
        "Alpha-Dir-TFT": {"base": alpha_dir_base, "rich": rich_common},
        "Alpha-Ret-TFT": {"base": alpha_ret_base, "rich": rich_common},
        "Risk-TFT": {"base": risk_base, "rich": rich_common},
        "Regime-Gate": {"base": regime_base, "rich": rich_common},
        "KeyLevel-Breakout-TFT": {"base": structure_base, "rich": rich_common},
        "RelativeStrength-Spread-TFT": {"base": relative_base, "rich": rich_common},
        "MicroStruct-Deriv-TFT": {"base": rich_common, "rich": rich_common},
        "OnChain-ETF-TFT": {"base": rich_common, "rich": rich_common},
        "Factor-Bridge-TFT": {"base": rich_common, "rich": rich_common},
        "Z-Combiner": {"base": base_ohlcv, "rich": []},
    }


def _synonym_map() -> dict:
    """同义映射：通用名 -> 实际列可能名称（按优先级）。"""
    return {
        "rsi": ["rsi", "rsi_14"],
        "macd_hist": ["macd_hist", "macd"],
        "adx": ["adx", "adx_scaled"],
        "boll_bandwidth": ["boll_bandwidth"],
        "ma5": ["ma5"],
        "ma20": ["ma20"],
        "obv": ["obv"],
        "atr": ["atr"],
        "cci": ["cci"],
        "momentum": ["momentum"],
        "ppo": ["ppo"],
        "stoch_rsi": ["stoch_rsi"],
        "williams_r": ["williams_r"],
        "tsi": ["tsi"],
        "donchian_high_20": ["donchian_high_20"],
        "donchian_low_20": ["donchian_low_20"],
        "keltner_upper_20": ["keltner_upper_20"],
        "keltner_lower_20": ["keltner_lower_20"],
        "squeeze_on": ["squeeze_on"],
        "trendline_slope_20": ["trendline_slope_20"],
        "ma_diff_5_20": ["ma_diff_5_20"],
        "boll_pctb": ["boll_pctb"],
        "amplitude_range": ["amplitude_range"],
        "rolling_volatility": ["rolling_volatility"],
        "parkinson_volatility": ["parkinson_volatility"],
        "ewmavolatility": ["ewmavolatility"],
        "return_skewness": ["return_skewness", "ret_skew_30d"],
        "atr_ratio_price": ["atr_ratio_price"],
        "atr_slope": ["atr_slope"],
        "var_95_30d": ["var_95_30d"],
        "cvar_95_30d": ["cvar_95_30d"],
        "ret_skew_30d": ["ret_skew_30d"],
        "ret_kurt_30d": ["ret_kurt_30d"],
        "atr_change": ["atr_change"],
        "atr_ratio_range": ["atr_ratio_range", "range_to_atr"],
        "trend_flag": ["trend_flag"],
        "range_flag": ["range_flag"],
        "high_vol_flag": ["high_vol_flag"],
        "stress_flag": ["stress_flag"],
        "volume_profile_hvn": ["volume_profile_hvn"],
        "volume_profile_lvn": ["volume_profile_lvn"],
        "volume_relative_20": ["volume_relative_20"],
        "volume_percentile": ["volume_percentile"],
        "support_level_20": ["support_level_20"],
        "resistance_level_20": ["resistance_level_20"],
        "breakout_recent_high": ["breakout_recent_high"],
        "pullback_from_high": ["pullback_from_high"],
        "distance_to_support": ["distance_to_support"],
        "distance_to_resistance": ["distance_to_resistance"],
        "volume_spike_near_support": ["volume_spike_near_support"],
        "volume_spike_near_resist": ["volume_spike_near_resist"],
        "tail_bias_norm": ["tail_bias_norm"],
        "r_sym": ["r_sym"],
        "r_btc": ["r_btc"],
        "beta_btc_60d": ["beta_btc_60d"],
        "price_position_ma200": ["price_position_ma200"],
        # rich
        "funding_rate": ["funding_rate", "funding_rate_z"],
        "oi_total": ["oi_total", "oi"],
        "taker_buy_sell_imbalance": ["taker_buy_sell_imbalance", "taker_imbalance_z"],
        "basis_z": ["basis_z", "basis"],
        "etf_flow": ["etf_flow", "etf_net_flow"],
        "etf_premium": ["etf_premium", "etf_premium_z"],
        "active_addresses": ["active_addresses", "active_addr_z"],
        "new_addresses": ["new_addresses"],
        "stablecoin_mcap": ["stablecoin_mcap"],
        "exch_netflow": ["exch_netflow"],
        "cb_premium": ["cb_premium"],
    }


def _resolve_and_report(df: pd.DataFrame, expert_map: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    syn = _synonym_map()
    resolved: dict = {}
    missing_rows = []
    resolved_rows = []
    cols = set(df.columns)
    for expert, ch_map in expert_map.items():
        resolved[expert] = {}
        for channel, wanted in ch_map.items():
            final_list = []
            for name in wanted:
                candidates = [name] + syn.get(name, [])
                found = None
                for cand in candidates:
                    if cand in cols:
                        found = cand
                        break
                if found is not None:
                    final_list.append(found)
                    resolved_rows.append({"expert": expert, "channel": channel, "source": name, "resolved": found})
                else:
                    missing_rows.append({"expert": expert, "channel": channel, "missing": name})
            resolved[expert][channel] = sorted(list(dict.fromkeys(final_list)))
    resolved_df = pd.DataFrame(resolved_rows, columns=["expert", "channel", "source", "resolved"]).sort_values(["expert", "channel", "source"]).reset_index(drop=True)
    missing_df = pd.DataFrame(missing_rows, columns=["expert", "channel", "missing"]).sort_values(["expert", "channel", "missing"]).reset_index(drop=True)
    return resolved, resolved_df, missing_df


def main():
    ap = argparse.ArgumentParser(description="Feature screening pipeline: tree+perm -> aggregate -> core set")
    ap.add_argument("--periods", type=str, default=None)
    ap.add_argument("--val-mode", type=str, default="ratio", choices=["ratio","days"]) 
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--preview", type=int, default=10)
    ap.add_argument("--tree-out", type=str, default="reports/feature_evidence/tree_perm")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--tft-gating", type=str, default=None)
    ap.add_argument("--tft-bonus", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--pkl", type=str, default=None, help="Override merged dataset PKL path")
    ap.add_argument("--time-perm", action="store_true")
    ap.add_argument("--block-len", type=int, default=36)
    ap.add_argument("--perm-repeats", type=int, default=5)
    ap.add_argument("--group-cols", type=str, default="symbol")
    ap.add_argument("--topk-per-pair", type=int, default=0)
    ap.add_argument("--min-appear-rate", type=float, default=0.0)
    ap.add_argument("--min-appear-rate-era", type=float, default=0.0)
    ap.add_argument("--out-summary", type=str, default="reports/feature_evidence/aggregated_core.csv")
    ap.add_argument("--out-allowlist", type=str, default="configs/selected_features.txt")
    ap.add_argument("--allowlist", type=str, default=None, help="Optional feature allowlist (one name per line)")
    # optional chained stages
    ap.add_argument("--with-filter", action="store_true")
    ap.add_argument("--with-embedded", action="store_true")
    ap.add_argument("--expert-name", type=str, default="generic")
    ap.add_argument("--channel", type=str, default="base")
    ap.add_argument("--boost-out-core", type=str, default="reports/feature_evidence/features_core.yaml")
    ap.add_argument("--boost-out-yaml", type=str, default="reports/feature_evidence/features_boost.yaml")
    ap.add_argument("--boost-prefixes", type=str, default="onchain_,funding_,oi_,basis_,etf_")
    ap.add_argument("--boost-limit-per-pair", type=int, default=12)
    args = ap.parse_args()
    periods = args.periods.split(",") if args.periods else None
    if periods is None:
        ds = load_split(pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl", val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio, allowlist_path=args.allowlist)
        periods = ds.periods
    else:
        ds = load_split(pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl", val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio, allowlist_path=args.allowlist)
    gcols = [c.strip() for c in (args.group_cols or "").split(",") if c.strip()]
    # 解析并输出字段映射与缺失报表
    expert_defaults = _experts_default_field_map()
    resolved_map, resolved_df, missing_df = _resolve_and_report(pd.concat([ds.train, ds.val], ignore_index=True), expert_defaults)
    os.makedirs("reports/experts", exist_ok=True)
    resolved_df.to_csv("reports/experts/fields_resolved.csv", index=False)
    missing_df.to_csv("reports/experts/missing_fields.csv", index=False)

    # optional chained stages
    if args.with_filter:
        # 注入 pinned（根据 expert_name/channel 使用解析后的列表）
        pinned_list = None
        if args.expert_name in resolved_map:
            pinned_list = resolved_map[args.expert_name].get(args.channel, [])
        params_dict = dict(args.__dict__)
        params = {}
        if params_dict:
            params = {k:v for k,v in params_dict.items() if k in ()}
        filt_params = {}
        if pinned_list:
            filt_params["pinned_features"] = tuple(pinned_list)
        run_filter_for_channel(
            expert_name=args.expert_name,
            channel=args.channel,
            pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl",
            val_mode=args.val_mode,
            val_days=args.val_days,
            val_ratio=args.val_ratio,
            allowlist_path=args.allowlist,
            params_dict={**filt_params},
        )
    if args.with_embedded:
        run_embedded_for_channel(
            expert_name=args.expert_name,
            channel=args.channel,
            pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl",
            val_mode=args.val_mode,
            val_days=args.val_days,
            val_ratio=args.val_ratio,
            allowlist_path=args.allowlist,
            targets_override=None,
            cfg={},
        )
    run_tree_perm(periods, args.val_mode, args.val_days, args.val_ratio, args.preview, args.tree_out, allowlist_path=args.allowlist, pkl_path=args.pkl, time_perm=bool(args.time_perm), perm_method="cyclic_shift", block_len=int(args.block_len), group_cols=gcols, repeats=int(args.perm_repeats))
    agg = aggregate_tree_perm(args.tree_out)
    core = build_unified_core(
        agg,
        args.weights,
        topk=args.topk,
        tft_file=args.tft_gating,
        tft_bonus=args.tft_bonus,
        topk_per_pair=(args.topk_per_pair if args.topk_per_pair and args.topk_per_pair > 0 else None),
        min_appear_rate=(args.min_appear_rate if args.min_appear_rate and args.min_appear_rate > 0 else None),
        min_appear_rate_era=(args.min_appear_rate_era if args.min_appear_rate_era and args.min_appear_rate_era > 0 else None),
    )
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    core.to_csv(args.out_summary, index=False)
    # 训练用清单：核心集 ∪ 当前专家通道的 pinned 默认集
    core_keep = core[core["keep"]]["feature"].astype(str).tolist()
    pinned_for_current = []
    try:
        pinned_for_current = (resolved_map.get(args.expert_name, {}) or {}).get(args.channel, []) or []
    except Exception:
        pinned_for_current = []
    union_feats = list(dict.fromkeys(list(core_keep) + list(pinned_for_current)))
    os.makedirs(os.path.dirname(args.out_allowlist), exist_ok=True)
    with open(args.out_allowlist, "w", encoding="utf-8") as f:
        for c in union_feats:
            f.write(f"{c}\n")
    try:
        extra_cnt = len(set(union_feats) - set(core_keep))
    except Exception:
        extra_cnt = 0
    print(f"[save] {args.out_allowlist} (core={len(core_keep)}, pinned_extra={extra_cnt}, total={len(union_feats)})")
    # export core/boost YAMLs
    boost_prefixes = [x.strip() for x in (args.boost_prefixes or "").split(",") if x.strip()]
    export_core_and_boost(core, agg, out_core_yaml=args.boost_out_core, out_boost_yaml=args.boost_out_yaml, boost_prefixes=boost_prefixes, boost_limit_per_pair=int(args.boost_limit_per_pair))
    print("[done] pipeline completed")


if __name__ == "__main__":
    main()
