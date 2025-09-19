from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


_REG_DEFAULT_WINDOW = {
    "fast": 24,
    "slow": 72,
}


def _safe_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def compute_regime_core_features(
    df: pd.DataFrame,
    group_cols: Sequence[str] = ("symbol", "period"),
    time_idx_col: str = "time_idx",
    close_col: str = "close",
    volume_col: str = "volume",
    funding_col: str | None = "funding_rate",
    oi_col: str | None = "open_interest",
    atr_col: str | None = "atr",
    high_col: str | None = "high",
    low_col: str | None = "low",
    window_fast: int = _REG_DEFAULT_WINDOW["fast"],
    window_slow: int = _REG_DEFAULT_WINDOW["slow"],
) -> pd.DataFrame:
    """Compute core regime features (volatility, momentum, slopes, structural gaps)."""
    if time_idx_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_idx_col}' column for alignment")

    work = df.copy()
    work = work.sort_values(list(group_cols) + [time_idx_col])

    results = []
    for keys, g in work.groupby(list(group_cols), sort=False):
        g = g.copy()
        g.set_index(time_idx_col, inplace=True)

        close = g[close_col] if close_col in g.columns else None
        returns = _safe_pct_change(close, periods=1) if close is not None else None
        if returns is None and "logreturn" in g.columns:
            returns = g["logreturn"].replace([np.inf, -np.inf], np.nan)

        realized_vol = returns.rolling(window_fast).std() if returns is not None else pd.Series(index=g.index, dtype=float)
        ewma_vol = returns.ewm(span=max(10, window_fast // 2)).std() if returns is not None else pd.Series(index=g.index, dtype=float)

        volume_change = _safe_pct_change(g[volume_col]) if volume_col in g.columns else pd.Series(index=g.index, dtype=float)

        funding_slope = g[funding_col].diff() if funding_col and funding_col in g.columns else pd.Series(index=g.index, dtype=float)
        oi_slope = g[oi_col].diff() if oi_col and oi_col in g.columns else pd.Series(index=g.index, dtype=float)

        momentum_fast = _safe_pct_change(close, periods=window_fast) if close is not None else pd.Series(index=g.index, dtype=float)
        momentum_slow = _safe_pct_change(close, periods=window_slow) if close is not None else pd.Series(index=g.index, dtype=float)

        atr_slope = g[atr_col].diff() if atr_col and atr_col in g.columns else pd.Series(index=g.index, dtype=float)

        if high_col in g.columns and low_col in g.columns and close is not None:
            rolling_high = g[high_col].rolling(window_slow).max()
            rolling_low = g[low_col].rolling(window_slow).min()
            structural_gap = (close - (rolling_high + rolling_low) / 2) / (close.replace(0, np.nan))
        else:
            structural_gap = pd.Series(index=g.index, dtype=float)

        out = pd.DataFrame({
            time_idx_col: g.index,
            "realized_vol": realized_vol,
            "ewma_vol": ewma_vol,
            "volume_change": volume_change,
            "funding_rate_slope": funding_slope,
            "open_interest_slope": oi_slope,
            "momentum_fast": momentum_fast,
            "momentum_slow": momentum_slow,
            "atr_slope": atr_slope,
            "structural_gap": structural_gap,
        })
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                out[col] = val
        else:
            out[group_cols[0]] = keys
        results.append(out)

    regime_df = pd.concat(results, ignore_index=True)
    regime_df.sort_values(list(group_cols) + [time_idx_col], inplace=True)
    regime_df.reset_index(drop=True, inplace=True)
    return regime_df
