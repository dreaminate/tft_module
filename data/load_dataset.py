import os
import pandas as pd
import pickle

class _NumpyCoreAliasUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def read_pickle_compat(path):
    import pandas as pd
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        if "numpy._core" not in str(e):
            raise
        with open(path, "rb") as fh:
            return _NumpyCoreAliasUnpickler(fh).load()
import numpy as np
from typing import List, Tuple, Optional, Dict
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, MultiNormalizer, TorchNormalizer
from torch.utils.data import DataLoader
from utils.feature_utils import get_pinned_features
import re


def _robust_parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a valid 'datetime' column by robust parsing.

    - Tries pandas>=2.0 format='mixed' with errors='coerce'
    - Falls back to generic to_datetime coercion
    - If still poor, tries using a numeric 'timestamp' (auto-detect ms vs s)
    """
    if 'datetime' in df.columns:
        s = df['datetime']
        parsed = None
        try:
            parsed = pd.to_datetime(s, format='mixed', errors='coerce')
        except TypeError:
            parsed = pd.to_datetime(s, errors='coerce', infer_datetime_format=False)
        if parsed.notna().any():
            df['datetime'] = parsed
            return df

    if 'timestamp' in df.columns:
        ts = pd.to_numeric(df['timestamp'], errors='coerce')
        if ts.notna().any():
            med = float(ts.dropna().median()) if ts.dropna().size else 0.0
            unit = 'ms' if med >= 1e11 else 's'
            parsed = pd.to_datetime(ts, unit=unit, errors='coerce')
            if parsed.notna().any():
                df['datetime'] = parsed
                return df

    # Last resort: coerce whatever is in 'datetime'
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        return df
    raise ValueError("Cannot construct a valid 'datetime' column from either 'datetime' or 'timestamp'.")


def get_dataloaders(
    data_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_mode: str = "days",
    val_days: int = 30,
    val_ratio: float = 0.2,
    targets_override: Optional[List[str]] = None,
    periods: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    selected_features_path: Optional[str] = None,
    pinned_features_cfg: Optional[Dict[str, List[str]]] = None,
    modality: Optional[str] = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, List[str], TimeSeriesDataSet, List[str], dict, dict]:
    # 优先使用传入的 data_path；若不可用则回退到默认 pkl 路径
    default_pkl = "data/pkl_merged/full_merged.pkl"
    use_path = data_path or default_pkl
    if not os.path.exists(use_path):
        use_path = default_pkl
    df = read_pickle_compat(use_path)

    df = df.astype({
        **{col: "float32" for col in df.select_dtypes(include="float64").columns},
        **{col: "int32" for col in df.select_dtypes(include="int64").columns},
    })
    df = _robust_parse_datetime(df)
    # 可选：按专家配置过滤周期与符号
    if periods:
        pset = set(str(p) for p in periods)
        df = df[df["period"].astype(str).isin(pset)]
    if symbols:
        sset = set(str(s) for s in symbols)
        df = df[df["symbol"].astype(str).isin(sset)]

    # 严格去掉分组键缺失的行，避免出现字符串 "nan" 被当成合法类别
    df = df[df["symbol"].notna() & df["period"].notna()].copy()
    # 规范化 symbol：去空格/统一大写/分隔符标准化；将 BTCUSDT 之类合并为 BTC_USDT
    def _canon_symbol(x: str) -> str:
        s = str(x).strip().upper()
        s = s.replace('/', '_').replace('-', '_')
        m = re.match(r"^([A-Z]+)(USDT|USD|BUSD|USDC)$", s)
        if m:
            s = f"{m.group(1)}_{m.group(2)}"
        return s
    try:
        df['symbol'] = df['symbol'].astype(str).map(_canon_symbol)
    except Exception:
        pass
    df = df.sort_values(["symbol", "period", "datetime"]).copy()
    df["time_idx"] = df.groupby(["symbol", "period"]).cumcount()
    df["symbol"] = df["symbol"].astype(str)
    df["period"] = df["period"].astype(str)

    numeric_cols = [
        c
        for c in df.columns
        if np.issubdtype(df[c].dtype, np.number)
        and c not in {"time_idx"}
    ]
    nan_fill_summary = {
        "total_nan_before": int(df[numeric_cols].isna().sum().sum()) if numeric_cols else 0,
        "zero_filled_columns": [],
    }
    if numeric_cols:
        df[numeric_cols] = df.groupby(["symbol", "period"], group_keys=False)[numeric_cols].ffill()
        df[numeric_cols] = df.groupby(["symbol", "period"], group_keys=False)[numeric_cols].bfill()
        remaining_nan = df[numeric_cols].isna().sum()
        nan_fill_summary["total_nan_after_ffill_bfill"] = int(remaining_nan.sum())
        still_missing_cols = remaining_nan[remaining_nan > 0].index.tolist()
        if still_missing_cols:
            df[still_missing_cols] = df[still_missing_cols].fillna(0.0)
            nan_fill_summary["zero_filled_columns"] = still_missing_cols
            nan_fill_summary["total_nan_after_fillna"] = int(df[numeric_cols].isna().sum().sum())
        else:
            nan_fill_summary["total_nan_after_fillna"] = int(remaining_nan.sum())
    else:
        nan_fill_summary["total_nan_after_ffill_bfill"] = 0
        nan_fill_summary["total_nan_after_fillna"] = 0

    # --- 全局时间划分 ---
    if val_mode == "days":
        val_cutoff_date = df["datetime"].max() - pd.Timedelta(days=val_days)
        df_train = df[df["datetime"] <= val_cutoff_date].copy()
        df_val = df[df["datetime"] > val_cutoff_date].copy()
    elif val_mode == "ratio":
        # 注意：对于时间序列，按比例的随机划分通常不推荐，因为它可能导致数据泄露。
        # 这里的实现是按时间顺序进行的，更安全。
        train_dfs, val_dfs = [], []
        for _, group in df.groupby(["symbol", "period"]):
            val_idx = int(len(group) * (1 - val_ratio))
            train_dfs.append(group.iloc[:val_idx])
            val_dfs.append(group.iloc[val_idx:])
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_val = pd.concat(val_dfs, ignore_index=True)
    else:
        raise ValueError(f"Unknown val_mode: {val_mode}")

    targets = [
        "target_binarytrend",
        "target_logreturn",
        "target_logsharpe_ratio",
        "target_breakout_count",
        "target_drawdown_prob",
        "target_max_drawdown",
        "target_pullback_prob",
        "target_return_outlier_lower_10",
        "target_return_outlier_upper_90",
        "target_sideway_detect",
        "target_trend_persistence",
    ]
    if targets_override:
        # 只保留数据中存在的列
        targets = [t for t in targets_override if t in df.columns]
    regression_targets = [
        "target_logreturn",
        "target_logsharpe_ratio",
        "target_breakout_count",
        "target_max_drawdown",
        "target_trend_persistence",
    ]

    known_reals = ["time_idx"]
    df.replace({None: np.nan}, inplace=True)


    # 候选未知实数特征（在筛选/拼接前的全量候选）
    unknown_reals = [
        c for c in df_train.columns
        if c not in known_reals
        and c not in ["timestamp", "symbol", "time_idx", "datetime", "period"]
        and not c.startswith("future")
        # 不把任何 target_* 列当作特征，只允许在 targets 列表中使用
        and not c.startswith("target_")
        and not df_train[c].isna().all()
    ]
    unknown_reals_before = list(unknown_reals)
    sel_path = selected_features_path or os.path.join("configs", "selected_features.txt")
    
    # 动态筛选的 Alpha 特征（来自 selected_features.txt）
    allowlist = []
    if sel_path and os.path.exists(sel_path):
        with open(sel_path, "r", encoding="utf-8") as fh:
            allowlist = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
    # 记录所用路径，便于排查
    sel_path_used = sel_path if sel_path and os.path.exists(sel_path) else None

    # 从配置注入的 Pinned 静态特征（数据集配置中的默认字段）
    pinned_features = []
    if pinned_features_cfg and modality:
        pinned_features = pinned_features_cfg.get(modality, [])
        # rich 和 comprehensive 自动继承 base
        if modality in ["rich", "comprehensive"]:
            base_pinned = pinned_features_cfg.get("base", [])
            pinned_features = list(dict.fromkeys(base_pinned + pinned_features))

    # 合并 allowlist 和 pinned features，去重
    combined_features = list(dict.fromkeys(allowlist + pinned_features))
    
    # 🔧 修复：始终执行特征过滤，确保只使用存在且有效的特征
    available_cols = set(df_train.columns)
    
    # 🎯 智能归一化特征匹配：根据当前周期推断正确的窗口后缀
    def _smart_feature_match(feature_list, available_cols, periods_in_data=None):
        """智能匹配特征，特别是归一化特征的窗口后缀"""
        matched_features = []
        period_window_map = {"1h": 96, "4h": 56, "1d": 30}
        
        # 如果有周期信息，确定当前主要周期的窗口大小
        dominant_windows = []
        if periods_in_data:
            for p in periods_in_data:
                if str(p) in period_window_map:
                    dominant_windows.append(period_window_map[str(p)])
        if not dominant_windows:
            dominant_windows = [96, 56, 30]  # 默认尝试所有窗口
        
        for feat in feature_list:
            if feat in available_cols:
                matched_features.append(feat)
            elif "_zn" in feat or "_mm" in feat:
                # 尝试智能匹配归一化特征
                if "_zn" in feat:
                    base_feat = feat.split("_zn")[0]
                    method = "_zn"
                else:
                    base_feat = feat.split("_mm")[0]
                    method = "_mm"
                
                found = False
                for window in dominant_windows:
                    candidate = f"{base_feat}{method}{window}"
                    if candidate in available_cols:
                        matched_features.append(candidate)
                        found = True
                        break
                
                if not found:
                    print(f"  ⚠️ 归一化特征 {feat} 未找到匹配项")
            else:
                print(f"  ❌ 特征不存在: {feat}")
        
        return matched_features
    
    if combined_features:
        # 使用配置指定的特征（智能匹配归一化特征）
        current_periods = [str(p) for p in periods] if periods else None
        final_features = _smart_feature_match(combined_features, available_cols, current_periods)
        unknown_reals = [c for c in unknown_reals if c in final_features]
        print(f"✅ 使用配置特征: {len(final_features)}/{len(combined_features)} 可用（含智能匹配）")
    else:
        # 🔧 修复：当没有配置特征时，使用所有可用的非全NaN特征（而不是跳过过滤）
        print("⚠️ 无特征配置，使用所有有效特征（去除全NaN列）")
        # 只保留非全NaN且数据质量好的特征
        valid_features = []
        for c in unknown_reals:
            if c in available_cols:
                col_data = df_train[c]
                # 检查列的有效性：非全NaN，且有效值比例>5%
                valid_ratio = col_data.notna().mean()
                if valid_ratio > 0.05:  # 至少5%的值是有效的
                    valid_features.append(c)
                else:
                    print(f"  ❌ 跳过列 {c}: 有效值比例仅 {valid_ratio:.1%}")
        
        unknown_reals = valid_features
        print(f"✅ 自动筛选有效特征: {len(unknown_reals)}/{len(unknown_reals_before)}")
    
    # 🧹 额外清理：移除内存中不需要的归一化列
    norm_cols_to_remove = []
    for col in df_train.columns:
        if ("_zn" in col or "_mm" in col) and col not in unknown_reals and col not in known_reals:
            norm_cols_to_remove.append(col)
    
    if norm_cols_to_remove:
        print(f"🧹 清理无用归一化列: {len(norm_cols_to_remove)} 个")
        df_train = df_train.drop(columns=norm_cols_to_remove, errors='ignore')
        df_val = df_val.drop(columns=norm_cols_to_remove, errors='ignore')
    # 汇总特征信息，用于训练阶段打印/落盘
    features_meta = {
        "selected_allowlist": allowlist,
        "selected_path": sel_path_used,
        "pinned_features": pinned_features,
        "combined_features": combined_features,
        "unknown_reals_before": unknown_reals_before,
        "unknown_reals_after": list(unknown_reals),
        "known_reals": list(known_reals),
        "static_categoricals": list(kwargs.get("static_categoricals", ["symbol", "period"]) or ["symbol", "period"]),
        "static_reals": list(kwargs.get("static_reals", []) or []),
        "nan_fill_summary": nan_fill_summary,
    }
    # 数据规模摘要
    def _vc(dff, col):
        try:
            return dff[col].astype(str).value_counts().to_dict()
        except Exception:
            return {}
    features_meta["dataset_summary"] = {
        "total_rows": int(len(df)),
        "train_rows": int(len(df_train)),
        "val_rows": int(len(df_val)),
        "train_by_symbol": _vc(df_train, "symbol"),
        "val_by_symbol": _vc(df_val, "symbol"),
        "train_by_period": _vc(df_train, "period"),
        "val_by_period": _vc(df_val, "period"),
    }

    symbol_classes = sorted(df_train["symbol"].dropna().unique().tolist())
    period_classes = sorted(df_train["period"].dropna().unique().tolist())
    sym2idx = {s: i for i, s in enumerate(symbol_classes)}
    per2idx = {p: i for i, p in enumerate(period_classes)}

    S, P, T = len(symbol_classes), len(period_classes), len(regression_targets)
    means = np.zeros((S, P, T), dtype="float32")
    stds = np.ones((S, P, T), dtype="float32")
    for (sym, per), g in df_train.groupby(["symbol", "period"]):
        si, pi = sym2idx[sym], per2idx[per]
        for ti, col in enumerate(regression_targets):
            if col in g.columns and len(g[col].dropna()):
                m = float(g[col].mean())
                s = float(g[col].std() or 0.0)
                stds[si, pi, ti] = max(s, 1e-8)
                means[si, pi, ti] = m

    norm_pack = {
        "regression_targets": regression_targets,
        "symbol_classes": symbol_classes,
        "period_classes": period_classes,
        "sym2idx": sym2idx,
        "per2idx": per2idx,
        "means": means,
        "stds": stds,
    }

    # No placeholder NaN class: we already drop rows with missing group keys.
    symbol_encoder = NaNLabelEncoder(add_nan=False)
    symbol_encoder.fit(pd.Series(symbol_classes))
    period_encoder = NaNLabelEncoder(add_nan=False)
    period_encoder.fit(pd.Series(period_classes))

    # 目标归一化器：单目标用单个 Normalizer，多目标用 MultiNormalizer
    if len(targets) == 1:
        target_normalizer = TorchNormalizer(method="identity", center=False)
        ts_target = targets[0]
    else:
        target_normalizer = MultiNormalizer([TorchNormalizer(method="identity", center=False)] * len(targets))
        # 单目标时使用字符串目标，多目标时使用列表
        ts_target = targets
    # 允许通过可选参数覆盖 TimeSeriesDataSet 关键配置
    max_encoder_length = int(kwargs.get("max_encoder_length", 36))
    max_prediction_length = int(kwargs.get("max_prediction_length", 1))
    add_relative_time = bool(kwargs.get("add_relative_time_idx", True))
    add_target_scales = bool(kwargs.get("add_target_scales", False))
    allow_missing = bool(kwargs.get("allow_missing_timesteps", True))
    static_cats = kwargs.get("static_categoricals", ["symbol", "period"]) or ["symbol", "period"]
    static_reals_cfg = kwargs.get("static_reals", []) or []

    ts_cfg = dict(
        time_idx="time_idx",
        target=ts_target,
        group_ids=["symbol", "period"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_cats,
        static_reals=static_reals_cfg,
        categorical_encoders={
            "symbol": symbol_encoder,
            "period": period_encoder,
        },
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=add_relative_time,
        add_target_scales=add_target_scales,
        allow_missing_timesteps=allow_missing,
        target_normalizer=target_normalizer,
    )

    train_ds = TimeSeriesDataSet(df_train, **ts_cfg)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, stop_randomization=True)

    train_loader = train_ds.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
    )
    # 验证集不要丢弃最后一个不满批次，避免样本过少时度量器无样本
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
    )
    return train_loader, val_loader, targets, train_ds, period_classes, norm_pack, features_meta
