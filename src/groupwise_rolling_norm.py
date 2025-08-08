# groupwise_rolling_norm.py
import pandas as pd
import numpy as np
from typing import List

def groupwise_rolling_norm(
    df: pd.DataFrame,
    cols: List[str],
    group_cols: List[str] = ("symbol", "period"),
    window: int = 48,
    methods: List[str] | None = None,           # <- 避免可变默认
    suffix_policy: str = "with_window",         # "with_window" or "plain"
    min_periods: int | None = None,             # <- 新增：更稳
    eps: float = 1e-8,                          # <- 新增：防 0
    clip_z: float | None = 5.0,                 # <- 新增：可选裁剪
    clip_mm: tuple[float, float] | None = (0.0, 1.0),
):
    """
    分组滑动归一化（不改原始列）：
      - z : (x - rolling_mean) / rolling_std
      - mm: (x - rolling_min)  / (rolling_max - rolling_min)
    注：所有 rolling 都先 shift(1) 再滚动，避免泄露。
    """
    if methods is None:
        methods = ["z", "mm"]
    if min_periods is None:
        min_periods = max(1, window // 4)

    df = df.copy()
    wtag = str(window) if suffix_policy == "with_window" else ""

    g = df.groupby(list(group_cols), sort=False)

    if "z" in methods:
        for c in cols:
            s = g[c].transform(lambda s: s.shift(1))
            rmean = s.rolling(window, min_periods=min_periods).mean()
            rstd  = s.rolling(window, min_periods=min_periods).std()
            z = (df[c] - rmean) / (rstd.abs() + eps)
            if clip_z is not None:
                z = z.clip(-clip_z, clip_z)
            df[f"{c}_zn{wtag}"] = z

    if "mm" in methods:
        for c in cols:
            s = g[c].transform(lambda s: s.shift(1))
            rmin = s.rolling(window, min_periods=min_periods).min()
            rmax = s.rolling(window, min_periods=min_periods).max()
            mm = (df[c] - rmin) / ((rmax - rmin).abs() + eps)
            if clip_mm is not None:
                lo, hi = clip_mm
                mm = mm.clip(lo, hi)
            df[f"{c}_mm{wtag}"] = mm

    return df
