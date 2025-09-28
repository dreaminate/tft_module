import pandas as pd
from typing import List, Dict

def _experts_default_field_map() -> Dict[str, Dict[str, List[str]]]:
    """基于 experts.md 推荐，给出每位专家的默认字段（通用名）。"""
    base_ohlcv = ["open", "high", "low", "close", "volume", "vwap"]
    alpha_dir_base = base_ohlcv + [
        "rsi", "macd_hist", "adx", "adx_scaled", "boll_bandwidth", "ma5", "ma20", "obv", "atr", "cci",
        "momentum", "ppo", "stoch_rsi", "williams_r", "tsi", "donchian_high_20", "donchian_low_20",
        "keltner_upper_20", "keltner_lower_20", "squeeze_on", "trendline_slope_20", "ma_diff_5_20",
        "boll_pctb", "amplitude_range",
    ]
    alpha_ret_base = alpha_dir_base + [
        "rolling_volatility", "parkinson_volatility", "ewmavolatility", "return_skewness",
        "atr_ratio_price", "atr_slope",
    ]
    risk_base = base_ohlcv + [
        "var_95_30d", "cvar_95_30d", "ret_skew_30d", "ret_kurt_30d", "atr", "atr_change", "atr_ratio_range",
    ]
    regime_base = base_ohlcv + [
        "trend_flag", "range_flag", "high_vol_flag", "stress_flag", "volume_profile_hvn", "volume_profile_lvn",
        "volume_relative_20", "volume_percentile",
    ]
    structure_base = base_ohlcv + [
        "support_level_20", "resistance_level_20", "breakout_recent_high", "pullback_from_high",
        "distance_to_support", "distance_to_resistance", "volume_spike_near_support", "volume_spike_near_resist",
        "tail_bias_norm", "amplitude_range",
    ]
    relative_base = base_ohlcv + ["r_sym", "r_btc", "beta_btc_60d", "price_position_ma200"]
    rich_common = [
        "funding_rate", "oi_total", "taker_buy_sell_imbalance", "basis_z", "etf_flow", "etf_premium",
        "active_addresses", "new_addresses", "stablecoin_mcap", "exch_netflow", "cb_premium",
    ]
    comprehensive_common = rich_common

    return {
        "Alpha-Dir-TFT": {"base": alpha_dir_base, "rich": rich_common, "comprehensive": comprehensive_common},
        "Alpha-Ret-TFT": {"base": alpha_ret_base, "rich": rich_common, "comprehensive": comprehensive_common},
        "Risk-TFT": {"base": risk_base, "rich": rich_common, "comprehensive": comprehensive_common},
        "Regime-Gate": {"base": regime_base, "rich": rich_common, "comprehensive": comprehensive_common},
        "KeyLevel-Breakout-TFT": {"base": structure_base, "rich": rich_common, "comprehensive": comprehensive_common},
        "RelativeStrength-Spread-TFT": {"base": relative_base, "rich": rich_common, "comprehensive": comprehensive_common},
        "MicroStruct-Deriv-TFT": {"base": rich_common, "rich": rich_common, "comprehensive": rich_common},
        "OnChain-ETF-TFT": {"base": rich_common, "rich": rich_common, "comprehensive": rich_common},
        "Factor-Bridge-TFT": {"base": rich_common, "rich": rich_common, "comprehensive": rich_common},
        "Z-Combiner": {"base": base_ohlcv, "rich": [], "comprehensive": []},
    }

def _synonym_map() -> Dict[str, List[str]]:
    """同义映射：通用名 -> 实际列可能名称（按优先级）。"""
    return {
        "rsi": ["rsi", "rsi_14"], "macd_hist": ["macd_hist", "macd"], "adx": ["adx", "adx_scaled"],
        "boll_bandwidth": ["boll_bandwidth"], "ma5": ["ma5"], "ma20": ["ma20"], "obv": ["obv"], "atr": ["atr"],
        "cci": ["cci"], "momentum": ["momentum"], "ppo": ["ppo"], "stoch_rsi": ["stoch_rsi"],
        "williams_r": ["williams_r"], "tsi": ["tsi"], "donchian_high_20": ["donchian_high_20"],
        "donchian_low_20": ["donchian_low_20"], "keltner_upper_20": ["keltner_upper_20"],
        "keltner_lower_20": ["keltner_lower_20"], "squeeze_on": ["squeeze_on"],
        "trendline_slope_20": ["trendline_slope_20"], "ma_diff_5_20": ["ma_diff_5_20"],
        "boll_pctb": ["boll_pctb"], "amplitude_range": ["amplitude_range"],
        "rolling_volatility": ["rolling_volatility"], "parkinson_volatility": ["parkinson_volatility"],
        "ewmavolatility": ["ewmavolatility"], "return_skewness": ["return_skewness", "ret_skew_30d"],
        "atr_ratio_price": ["atr_ratio_price"], "atr_slope": ["atr_slope"], "var_95_30d": ["var_95_30d"],
        "cvar_95_30d": ["cvar_95_30d"], "ret_skew_30d": ["ret_skew_30d"], "ret_kurt_30d": ["ret_kurt_30d"],
        "atr_change": ["atr_change"], "atr_ratio_range": ["atr_ratio_range", "range_to_atr"],
        "trend_flag": ["trend_flag"], "range_flag": ["range_flag"], "high_vol_flag": ["high_vol_flag"],
        "stress_flag": ["stress_flag"], "volume_profile_hvn": ["volume_profile_hvn"],
        "volume_profile_lvn": ["volume_profile_lvn"], "volume_relative_20": ["volume_relative_20"],
        "volume_percentile": ["volume_percentile"], "support_level_20": ["support_level_20"],
        "resistance_level_20": ["resistance_level_20"], "breakout_recent_high": ["breakout_recent_high"],
        "pullback_from_high": ["pullback_from_high"], "distance_to_support": ["distance_to_support"],
        "distance_to_resistance": ["distance_to_resistance"],
        "volume_spike_near_support": ["volume_spike_near_support"],
        "volume_spike_near_resist": ["volume_spike_near_resist"], "tail_bias_norm": ["tail_bias_norm"],
        "r_sym": ["r_sym"], "r_btc": ["r_btc"], "beta_btc_60d": ["beta_btc_60d"],
        "price_position_ma200": ["price_position_ma200"],
        "funding_rate": ["funding_rate", "funding_rate_z"], "oi_total": ["oi_total", "oi"],
        "taker_buy_sell_imbalance": ["taker_buy_sell_imbalance", "taker_imbalance_z"],
        "basis_z": ["basis_z", "basis"], "etf_flow": ["etf_flow", "etf_net_flow"],
        "etf_premium": ["etf_premium", "etf_premium_z"], "active_addresses": ["active_addresses", "active_addr_z"],
        "new_addresses": ["new_addresses"], "stablecoin_mcap": ["stablecoin_mcap"],
        "exch_netflow": ["exch_netflow"], "cb_premium": ["cb_premium"],
    }

def get_pinned_features(expert_name: str, channel: str, all_columns: List[str]) -> List[str]:
    """
    Resolves the pinned features for a given expert and channel.
    It resolves synonym names to actual column names found in the dataframe.
    For 'rich' and 'comprehensive' channels, it automatically includes the 'base' features.
    """
    expert_map = _experts_default_field_map()
    syn = _synonym_map()
    
    if not expert_name or expert_name not in expert_map:
        return []
        
    channel = channel or "base"
    wanted = expert_map[expert_name].get(channel, [])
    
    if channel in ['rich', 'comprehensive']:
        base_features = expert_map[expert_name].get('base', [])
        wanted = list(dict.fromkeys(base_features + wanted))

    resolved_list = []
    available_columns = set(all_columns)
    for name in wanted:
        candidates = [name] + syn.get(name, [])
        found_feature = None
        for cand in candidates:
            if cand in available_columns:
                found_feature = cand
                break
        if found_feature:
            resolved_list.append(found_feature)
            
    return sorted(list(dict.fromkeys(resolved_list)))
