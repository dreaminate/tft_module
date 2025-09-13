from __future__ import annotations
from typing import Dict, Iterable

def filter_weights_by_period(weights: Dict[str, float], periods: Iterable[str]) -> Dict[str, float]:
    period_set = set(periods)
    filtered = {k: v for k, v in weights.items() if k.split("@")[-1] in period_set or k == "val_loss_for_ckpt"}
    removed = set(weights) - set(filtered)
    if removed:
        print(f"[⚠️ Warning] Removing metrics for unsupported periods: {sorted(removed)}")
    return filtered

