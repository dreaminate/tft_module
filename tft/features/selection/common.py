import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


TARGETS = [
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

REGRESSION_TARGETS = [
    "target_logreturn",
    "target_logsharpe_ratio",
    "target_breakout_count",
    "target_max_drawdown",
    "target_trend_persistence",
]


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    features: List[str]
    targets: List[str]
    periods: List[str]


def _read_df_prefer_pkl(pkl_path: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    if os.path.exists(pkl_path):
        # 兼容 numpy._core pickle
        import pickle
        class _NumpyCoreAliasUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("numpy._core"):
                    module = module.replace("numpy._core", "numpy.core")
                return super().find_class(module, name)
        with open(pkl_path, "rb") as fh:
            df = _NumpyCoreAliasUnpickler(fh).load()
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{pkl_path} does not contain a pandas DataFrame")
        return df
    if csv_path and os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither {pkl_path} nor {csv_path} found")


def discover_features(df: pd.DataFrame, allowlist_path: Optional[str] = None) -> List[str]:
    known_reals = ["time_idx"]  # 如果未来需要可扩展
    # 推断数值特征：排除非数值与辅助列、目标列
    candidates = [
        c for c in df.columns
        if c not in known_reals
        and c not in ("timestamp", "symbol", "time_idx", "datetime", "period")
        and c not in TARGETS
        and not str(c).startswith("future")
    ]
    # 仅保留非全缺失列
    candidates = [c for c in candidates if not df[c].isna().all()]

    # 若存在白名单，则过滤
    if allowlist_path and os.path.exists(allowlist_path):
        with open(allowlist_path, "r", encoding="utf-8") as fh:
            allow = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
        before = len(candidates)
        candidates = [c for c in candidates if c in allow]
        print(f"[FS] apply selected_features.txt: {before} -> {len(candidates)}")
    return candidates


def split_train_val(
    df: pd.DataFrame,
    val_mode: str = "ratio",
    val_days: int = 30,
    val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])  # 容错
    df = df.sort_values(["symbol", "period", "datetime"]).copy()
    train_parts, val_parts = [], []
    for (_, _), g in df.groupby(["symbol", "period"], sort=False):
        g = g.sort_values("datetime")
        if val_mode == "days":
            cutoff = g["datetime"].max() - pd.Timedelta(days=int(val_days))
            train_parts.append(g[g["datetime"] <= cutoff])
            val_parts.append(g[g["datetime"] > cutoff])
        else:
            idx = int(len(g) * (1 - float(val_ratio)))
            train_parts.append(g.iloc[:idx])
            val_parts.append(g.iloc[idx:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
    )


def load_split(
    pkl_path: str = "data/pkl_merged/full_merged.pkl",
    csv_path: str = "data/merged/full_merged.csv",
    val_mode: str = "ratio",
    val_days: int = 30,
    val_ratio: float = 0.2,
    allowlist_path: Optional[str] = None,
) -> DatasetSplit:
    df = _read_df_prefer_pkl(pkl_path, csv_path)
    # 类型与排序
    df["symbol"] = df["symbol"].astype(str)
    df["period"] = df["period"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"])  # 容错
    df = df.sort_values(["symbol", "period", "datetime"]).copy()

    features = discover_features(df, allowlist_path)
    train_df, val_df = split_train_val(df, val_mode=val_mode, val_days=val_days, val_ratio=val_ratio)
    periods = sorted(df["period"].dropna().astype(str).unique().tolist())
    return DatasetSplit(train=train_df, val=val_df, features=features, targets=TARGETS, periods=periods)


def is_classification_target(t: str) -> bool:
    return t not in REGRESSION_TARGETS


def safe_numeric_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            # 强制转换为数值，失败置 NaN
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

