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
try:
    from tqdm.auto import tqdm
except Exception:  # fallback if tqdm unavailable
    def tqdm(x, **kwargs):
        return x


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


def fit_tree_and_permutation(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    is_cls: bool,
    n_estimators: int = 300,
    random_state: int = 42,
    time_perm: bool = False,
    val_df: pd.DataFrame | None = None,
    perm_method: str = "cyclic_shift",
    block_len: int = 36,
    group_cols: List[str] | None = None,
    repeats: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if is_cls:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=random_state)
        y_tr_bin = y_tr.fillna(0).clip(0, 1).astype(int)
        y_va_bin = y_va.fillna(0).clip(0, 1).astype(int)
        clf.fit(X_tr, y_tr_bin)
        prob = clf.predict_proba(X_va)[:, 1]
        base_score = f1_score(y_va_bin, (prob >= 0.5).astype(int))
        if not time_perm:
            perm = permutation_importance(clf, X_va, y_va_bin, scoring="f1", n_repeats=10, n_jobs=1, random_state=random_state)
            return clf.feature_importances_, perm.importances_mean, perm.importances_std, float(base_score)
        # time-aware permutation
        assert val_df is not None, "val_df required for time-aware permutation"
        cols = list(X_va.columns)
        means, stds = [], []
        # precompute groups of indices
        gcols = group_cols or ["symbol"]
        if not all(c in val_df.columns for c in gcols):
            gcols = [c for c in gcols if c in val_df.columns] or ["symbol"]
        group_index = {k: idx.index.to_numpy() for k, idx in val_df.groupby(gcols, sort=False).groups.items()}
        def _score(Xtmp: pd.DataFrame) -> float:
            p = clf.predict_proba(Xtmp)[:, 1]
            return f1_score(y_va_bin, (p >= 0.5).astype(int))
        base = float(base_score)
        for j, c in enumerate(cols):
            deltas = []
            for r in range(int(repeats)):
                Xp = X_va.copy()
                arr = Xp[c].to_numpy()
                for _, idx in group_index.items():
                    if len(idx) <= 1:
                        continue
                    # choose shift >= block_len but < len(idx)
                    max_shift = max(1, len(idx) - 1)
                    s = block_len if block_len <= max_shift else min(max_shift, max(1, len(idx) // 3))
                    if s >= max_shift:
                        s = max_shift
                    # cyclic roll within group
                    arr[idx] = np.roll(arr[idx], s)
                Xp[c] = arr
                sc = _score(Xp)
                deltas.append(base - float(sc))
            means.append(float(np.mean(deltas)))
            stds.append(float(np.std(deltas)))
        return clf.feature_importances_, np.array(means), np.array(stds), float(base_score)
    else:
        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=None, n_jobs=-1, random_state=random_state)
        reg.fit(X_tr, y_tr)
        pred = reg.predict(X_va)
        try:
            rmse = mean_squared_error(y_va, pred, squared=False)
        except TypeError:
            rmse = np.sqrt(mean_squared_error(y_va, pred))
        base_score = -float(rmse)
        if not time_perm:
            # older sklearn may not support neg_root_mean_squared_error
            perm = permutation_importance(reg, X_va, y_va, scoring="neg_mean_squared_error", n_repeats=10, n_jobs=1, random_state=random_state)
            return reg.feature_importances_, perm.importances_mean, perm.importances_std, float(base_score)
        # time-aware permutation
        assert val_df is not None, "val_df required for time-aware permutation"
        cols = list(X_va.columns)
        means, stds = [], []
        gcols = group_cols or ["symbol"]
        if not all(c in val_df.columns for c in gcols):
            gcols = [c for c in gcols if c in val_df.columns] or ["symbol"]
        group_index = {k: idx.index.to_numpy() for k, idx in val_df.groupby(gcols, sort=False).groups.items()}
        def _score(Xtmp: pd.DataFrame) -> float:
            p = reg.predict(Xtmp)
            try:
                rm = float(mean_squared_error(y_va, p, squared=False))
            except TypeError:
                rm = float(np.sqrt(mean_squared_error(y_va, p)))
            return -rm
        base = float(base_score)
        for j, c in enumerate(cols):
            deltas = []
            for r in range(int(repeats)):
                Xp = X_va.copy()
                arr = Xp[c].to_numpy()
                for _, idx in group_index.items():
                    if len(idx) <= 1:
                        continue
                    max_shift = max(1, len(idx) - 1)
                    s = block_len if block_len <= max_shift else min(max_shift, max(1, len(idx) // 3))
                    if s >= max_shift:
                        s = max_shift
                    arr[idx] = np.roll(arr[idx], s)
                Xp[c] = arr
                sc = _score(Xp)
                deltas.append(base - float(sc))
            means.append(float(np.mean(deltas)))
            stds.append(float(np.std(deltas)))
        return reg.feature_importances_, np.array(means), np.array(stds), float(base_score)


def run(
    periods: List[str],
    val_mode: str,
    val_days: int,
    val_ratio: float,
    topn_preview: int,
    out_dir: str,
    allowlist_path: str | None = None,
    pkl_path: str | None = None,
    time_perm: bool = False,
    perm_method: str = "cyclic_shift",
    block_len: int = 36,
    block_len_by_period: Dict[str, int] | None = None,
    group_cols: List[str] | None = None,
    repeats: int = 5,
    targets_override: List[str] | None = None,
) -> None:
    ds = load_split(
        pkl_path=pkl_path or "data/pkl_merged/full_merged.pkl",
        val_mode=val_mode,
        val_days=val_days,
        val_ratio=val_ratio,
        allowlist_path=allowlist_path,
        targets_override=targets_override,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[tree_perm] periods={periods} targets={ds.targets}")
    for per in tqdm(periods, desc="periods"):
        per_tr = ds.train[ds.train["period"].astype(str) == str(per)]
        per_va = ds.val[ds.val["period"].astype(str) == str(per)]
        if per_tr.empty or per_va.empty:
            print(f"[warn] period={per} has empty train/val; skip")
            continue
        per_block_len = int(block_len_by_period.get(str(per), block_len)) if isinstance(block_len_by_period, dict) else int(block_len)
        per_dir = Path(out_dir) / str(per)
        per_dir.mkdir(parents=True, exist_ok=True)
        for target in tqdm(ds.targets, desc=f"targets@{per}"):
            if target not in per_tr.columns:
                continue
            is_cls = target not in ("target_logreturn","target_logsharpe_ratio","target_breakout_count","target_max_drawdown","target_trend_persistence")
            X_tr, y_tr, X_va, y_va, _ = _prep_xy(per_tr, per_va, ds.features, target)
            importances, p_mean, p_std, base = fit_tree_and_permutation(
                X_tr, y_tr, X_va, y_va,
                is_cls=is_cls,
                time_perm=time_perm,
                val_df=per_va,
                perm_method=perm_method,
                block_len=per_block_len,
                group_cols=group_cols,
                repeats=repeats,
            )
            out = pd.DataFrame({
                "feature": X_tr.columns,
                "tree_importance": importances,
                "perm_mean": p_mean,
                "perm_std": p_std,
                "base_score": float(base),
            }).sort_values(["tree_importance","perm_mean"], ascending=[False, False])
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
    ap.add_argument("--allowlist", type=str, default=None, help="Optional feature allowlist (one name per line)")
    ap.add_argument("--pkl", type=str, default=None, help="Override merged dataset PKL path")
    ap.add_argument("--time-perm", action="store_true", help="Use time-aware permutation (cyclic shift)")
    ap.add_argument("--block-len", type=int, default=36)
    ap.add_argument("--group-cols", type=str, default="symbol", help="Comma-separated group cols for permutation (e.g., symbol,period)")
    ap.add_argument("--perm-repeats", type=int, default=5)
    ap.add_argument("--targets", type=str, default=None, help="Comma-separated targets to include (override)")
    args = ap.parse_args()
    targets_override = [t.strip() for t in (args.targets or '').split(',') if t.strip()] or None
    ds = load_split(
        pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl",
        val_mode=args.val_mode,
        val_days=args.val_days,
        val_ratio=args.val_ratio,
        allowlist_path=args.allowlist,
        targets_override=targets_override,
    )
    periods = args.periods.split(",") if args.periods else ds.periods
    gcols = [c.strip() for c in (args.group_cols or "").split(",") if c.strip()]
    run(
        periods,
        args.val_mode,
        args.val_days,
        args.val_ratio,
        args.preview,
        args.out,
        allowlist_path=args.allowlist,
        pkl_path=args.pkl,
        time_perm=bool(args.time_perm),
        perm_method="cyclic_shift",
        block_len=int(args.block_len),
        group_cols=gcols,
        repeats=int(args.perm_repeats),
        targets_override=targets_override,
    )


if __name__ == "__main__":
    main()
