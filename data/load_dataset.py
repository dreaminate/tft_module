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
from typing import List, Tuple, Optional
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder, MultiNormalizer, TorchNormalizer
from torch.utils.data import DataLoader


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
    **kwargs,
) -> Tuple[DataLoader, DataLoader, List[str], TimeSeriesDataSet, List[str], dict]:
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

    df = df.sort_values(["symbol", "period", "datetime"]).copy()
    df["time_idx"] = df.groupby(["symbol", "period"]).cumcount()
    df["symbol"], df["period"] = df["symbol"].astype(str), df["period"].astype(str)

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

    train_parts, val_parts = [], []
    for (sym, per), group_df in df.groupby(["symbol", "period"]):
        group_df = group_df.sort_values("datetime")
        if val_mode == "days":
            val_cutoff = group_df["datetime"].max() - pd.Timedelta(days=val_days)
            train_parts.append(group_df[group_df["datetime"] <= val_cutoff])
            val_parts.append(group_df[group_df["datetime"] > val_cutoff])
        elif val_mode == "ratio":
            val_idx = int(len(group_df) * (1 - val_ratio))
            train_parts.append(group_df.iloc[:val_idx])
            val_parts.append(group_df.iloc[val_idx:])

    df_train = pd.concat(train_parts, ignore_index=True)
    df_val = pd.concat(val_parts, ignore_index=True)

    unknown_reals = [
        c for c in df_train.columns
        if c not in known_reals
        and c not in ["timestamp", "symbol", "time_idx", "datetime", "period"]
        and not c.startswith("future")
        and c not in targets
        and not df_train[c].isna().all()
    ]
    sel_path = selected_features_path or os.path.join("configs", "selected_features.txt")
    if sel_path and os.path.exists(sel_path):
        with open(sel_path, "r", encoding="utf-8") as fh:
            allow = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
        unknown_reals = [c for c in unknown_reals if c in allow]

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

    symbol_encoder = NaNLabelEncoder(add_nan=True)
    symbol_encoder.fit(pd.Series(symbol_classes))
    period_encoder = NaNLabelEncoder(add_nan=True)
    period_encoder.fit(pd.Series(period_classes))

    # 目标归一化器：单目标用单个 Normalizer，多目标用 MultiNormalizer
    if len(targets) == 1:
        target_normalizer = TorchNormalizer(method="identity", center=False)
        ts_target = targets[0]
    else:
        target_normalizer = MultiNormalizer([TorchNormalizer(method="identity", center=False)] * len(targets))
        # 单目标时使用字符串目标，多目标时使用列表
        ts_target = targets
    ts_cfg = dict(
        time_idx="time_idx",
        target=ts_target,
        group_ids=["symbol", "period"],
        max_encoder_length=36,
        max_prediction_length=1,
        static_categoricals=["symbol", "period"],
        static_reals=[],
        categorical_encoders={
            "symbol": symbol_encoder,
            "period": period_encoder,
        },
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=False,
        allow_missing_timesteps=True,
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
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True,
    )
    return train_loader, val_loader, targets, train_ds, period_classes, norm_pack
