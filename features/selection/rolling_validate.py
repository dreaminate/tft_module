from __future__ import annotations
import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error

from .common import load_split, is_classification_target, safe_numeric_copy


def _evaluate(features: List[str], train_df: pd.DataFrame, val_df: pd.DataFrame, targets: List[str], weights_yaml: str) -> float:
    import yaml
    with open(weights_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ws = [float(x) for x in cfg.get("custom_weights", [1.0] * len(targets))]
    wmap = {t: float(w) for t, w in zip(targets, ws)}
    cols = [c for c in features if c in train_df.columns]
    if not cols:
        return -1e9
    Xtr = safe_numeric_copy(train_df[cols]).fillna(0).replace([np.inf, -np.inf], 0)
    Xva = safe_numeric_copy(val_df[cols]).fillna(0).replace([np.inf, -np.inf], 0)
    total, wsum = 0.0, 0.0
    for t in targets:
        if t not in train_df.columns:
            continue
        w = float(wmap.get(t, 1.0))
        if w <= 0:
            continue
        if is_classification_target(t):
            ytr = train_df[t].fillna(0).clip(0, 1).astype(int)
            yva = val_df[t].fillna(0).clip(0, 1).astype(int)
            est = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced_subsample", random_state=1)
            est.fit(Xtr, ytr); prob = est.predict_proba(Xva)[:, 1]
            sc = f1_score(yva, (prob >= 0.5).astype(int))
        else:
            ytr = train_df[t].astype(float)
            yva = val_df[t].astype(float)
            est = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=1)
            est.fit(Xtr, ytr); pred = est.predict(Xva)
            sc = -mean_squared_error(yva, pred, squared=False)
        total += w * float(sc); wsum += w
    if wsum <= 0:
        return -1e9
    return float(total / wsum)


def rolling_splits_period(df: pd.DataFrame, n_splits: int = 5, val_ratio: float = 0.2) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    g = df.copy().sort_values("datetime")
    times = g["datetime"].drop_duplicates().sort_values().to_numpy()
    if len(times) < 10:
        return []
    step = max(1, int(len(times) * (1 - val_ratio) / (n_splits + 1)))
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(1, n_splits + 1):
        cut_idx = i * step
        if cut_idx >= len(times) - 1:
            break
        cut_time = times[cut_idx]
        val_end_idx = min(len(times) - 1, cut_idx + int(len(times) * val_ratio))
        val_end_time = times[val_end_idx]
        tr = g[g["datetime"] <= cut_time]
        va = g[(g["datetime"] > cut_time) & (g["datetime"] <= val_end_time)]
        if not tr.empty and not va.empty:
            splits.append((tr, va))
    return splits


def main():
    ap = argparse.ArgumentParser(description="Rolling time-series validation for a selected feature set")
    ap.add_argument("--features", type=str, default="configs/selected_features.txt")
    ap.add_argument("--val-mode", type=str, default="ratio", choices=["ratio","days"]) 
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--weights", type=str, default="configs/weights_config.yaml")
    ap.add_argument("--periods", type=str, default=None)
    ap.add_argument("--pkl", type=str, default=None, help="Override merged dataset PKL path")
    args = ap.parse_args()
    with open(args.features, "r", encoding="utf-8") as f:
        feats = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    ds = load_split(pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl", val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio)
    periods = args.periods.split(",") if args.periods else ds.periods
    results: List[Dict[str, float]] = []
    for per in periods:
        dfp = ds.train[ds.train["period"].astype(str) == str(per)].copy()
        if dfp.empty:
            continue
        splits = rolling_splits_period(dfp, n_splits=args.splits, val_ratio=args.val_ratio)
        if not splits:
            print(f"[warn] not enough timesteps for period={per}; skipped")
            continue
        scores = []
        for (tr, va) in splits:
            sc = _evaluate(feats, tr, va, ds.targets, args.weights)
            scores.append(sc)
        if scores:
            mean_sc = float(np.mean(scores))
            results.append({"period": per, "mean_score": mean_sc, "folds": len(scores)})
            print(f"[roll] period={per} mean_score={mean_sc:.6f} folds={len(scores)}")
    if results:
        dfres = pd.DataFrame(results)
        out = "reports/feature_evidence/rolling_validate.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        dfres.to_csv(out, index=False)
        print(f"[save] {out}")
    else:
        print("[warn] no rolling results produced")


if __name__ == "__main__":
    main()
