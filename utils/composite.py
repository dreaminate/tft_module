"""Utility helpers for composite score weights."""
from __future__ import annotations
from typing import Dict, Iterable


def filter_weights_by_period(weights: Dict[str, float], periods: Iterable[str]) -> Dict[str, float]:
    """Filter composite weight dict to only include metrics for given periods.

    Parameters
    ----------
    weights: Dict[str, float]
        Original composite weight mapping.
    periods: Iterable[str]
        Period names (e.g., ["4h"]) that exist in the dataset.

    Returns
    -------
    Dict[str, float]
        Filtered weights containing only metrics whose trailing
        "@period" part matches one of ``periods``. Warns about removed keys.
    """
    period_set = set(periods)
    filtered = {k: v for k, v in weights.items() if k.split("@")[-1] in period_set or k == "val_loss_for_ckpt"}
    removed = set(weights) - set(filtered)
    if removed:
        print(f"[⚠️ Warning] Removing metrics for unsupported periods: {sorted(removed)}")
    return filtered