from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, mean_squared_error

from .common import load_split, safe_numeric_copy


def _prep_xy(train_df: pd.DataFrame, val_df: pd.DataFrame, features: List[str], target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, float]]:
    cols = [c for c in features if c in train_df.columns]
    X_tr = safe_numeric_copy(train_df[cols])
    X_va = safe_numeric_copy(val_df[cols])
    y_tr = train_df[target].astype(float)
    y_va = val_df[target].astype(float)
    med = X_tr.median(axis=0, numeric_only=True)
    X_tr = X_tr.fillna(med).replace([np.inf, -np.inf], 0)
    X_va = X_va.fillna(med).replace([np.inf, -np.inf], 0)
    return X_tr, y_tr, X_va, y_va, med.to_dict()


def fit_tree_and_permutation(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series, is_cls: bool, n_estimators: int = 300, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if is_cls:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=random_state)
        y_tr_bin = y_tr.fillna(0).clip(0, 1).astype(int)
        y_va_bin = y_va.fillna(0).clip(0, 1).astype(int)
        clf.fit(X_tr, y_tr_bin)
        prob = clf.predict_proba(X_va)[:, 1]
        base_score = f1_score(y_va_bin, (prob >= 0.5).astype(int))
        perm = permutation_importance(clf, X_va, y_va_bin, scoring="f1", n_repeats=10, n_jobs=-1, random_state=random_state)
        return clf.feature_importances_, perm.importances_mean, perm.importances_std, float(base_score)
    else:
        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=None, n_jobs=-1, random_state=random_state)
        reg.fit(X_tr, y_tr)
        pred = reg.predict(X_va)
        rmse = mean_squared_error(y_va, pred, squared=False)
        base_score = -float(rmse)
        perm = permutation_importance(reg, X_va, y_va, scoring="neg_root_mean_squared_error", n_repeats=10, n_jobs=-1, random_state=random_state)
        return reg.feature_importances_, perm.importances_mean, perm.importances_std, float(base_score)


def run(periods: List[str], val_mode: str, val_days: int, val_ratio: float, topn_preview: int, out_dir: str) -> None:
    ds = load_split(val_mode=val_mode, val_days=val_days, val_ratio=val_ratio)
    os.makedirs(out_dir, exist_ok=True)
    for per in periods:
        per_tr = ds.train[ds.train["period"].astype(str) == str(per)]
        per_va = ds.val[ds.val["period"].astype(str) == str(per)]
        if per_tr.empty or per_va.empty:
            print(f"[warn] period={per} has empty train/val; skip")
            continue
        per_dir = Path(out_dir) / str(per)
        per_dir.mkdir(parents=True, exist_ok=True)
        for target in ds.targets:
            if target not in per_tr.columns:
                continue
            is_cls = target not in ("target_logreturn","target_logsharpe_ratio","target_breakout_count","target_max_drawdown","target_trend_persistence")
            X_tr, y_tr, X_va, y_va, _ = _prep_xy(per_tr, per_va, ds.features, target)
            importances, p_mean, p_std, base = fit_tree_and_permutation(X_tr, y_tr, X_va, y_va, is_cls=is_cls)
            out = pd.DataFrame({"feature": X_tr.columns, "tree_importance": importances, "perm_mean": p_mean, "perm_std": p_std}).sort_values(["tree_importance","perm_mean"], ascending=[False, False])
            csv_path = per_dir / f"{target}_importances.csv"
            out.to_csv(csv_path, index=False)
            print(f"[save] {csv_path} | base_score={base:.6f}")
            if topn_preview > 0:
                print(out.head(topn_preview).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="Compute per-period & per-target feature importances (tree + permutation)")
    ap.add_argument("--periods", type=str, default=None, help="Comma-separated periods to include (default: all)")
    ap.add_argument("--val-mode", type=str, default="ratio", choices=["ratio","days"])
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--preview", type=int, default=10)
    ap.add_argument("--out", type=str, default="reports/feature_evidence/tree_perm")
    args = ap.parse_args()
    ds = load_split(val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio)
    periods = args.periods.split(",") if args.periods else ds.periods
    run(periods, args.val_mode, args.val_days, args.val_ratio, args.preview, args.out)


if __name__ == "__main__":
    main()

