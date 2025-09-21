from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, mean_squared_error

from .common import is_classification_target, load_split, safe_numeric_copy

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
    def tqdm(x, **_):
        return x


def _prep_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    target: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, float]]:
    cols = [c for c in features if c in train_df.columns]
    X_tr = safe_numeric_copy(train_df[cols])
    X_va = safe_numeric_copy(val_df[cols])
    y_tr = train_df[target].astype(float)
    y_va = val_df[target].astype(float)
    med = X_tr.median(axis=0, numeric_only=True)
    X_tr = X_tr.fillna(med).replace([np.inf, -np.inf], 0)
    X_va = X_va.fillna(med).replace([np.inf, -np.inf], 0)
    return X_tr, y_tr, X_va, y_va, med.to_dict()


def _build_group_index(val_df: pd.DataFrame, group_cols: Optional[List[str]]) -> Dict[str, np.ndarray]:
    if not group_cols:
        return {"__all__": val_df.index.to_numpy()}
    valid_cols = [c for c in group_cols if c in val_df.columns]
    if not valid_cols:
        return {"__all__": val_df.index.to_numpy()}
    groups = {}
    for key, idx in val_df.groupby(valid_cols, sort=False).groups.items():
        groups[str(key)] = np.asarray(idx)
    return groups or {"__all__": val_df.index.to_numpy()}


def _build_valid_mask(length: int, group_index: Dict[str, np.ndarray], embargo: int, purge: int) -> np.ndarray:
    mask = np.ones(length, dtype=bool)
    drop_front = max(0, int(embargo))
    drop_back = max(0, int(purge))
    if drop_front == 0 and drop_back == 0:
        return mask
    for idx in group_index.values():
        if drop_front > 0:
            take = min(drop_front, len(idx))
            mask[idx[:take]] = False
        if drop_back > 0:
            take = min(drop_back, len(idx))
            mask[idx[-take:]] = False
    return mask


def fit_tree_and_permutation(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    *,
    is_cls: bool,
    n_estimators: int = 300,
    random_state: int = 42,
    time_perm: bool = False,
    val_df: Optional[pd.DataFrame] = None,
    perm_method: str = "cyclic_shift",
    block_len: int = 36,
    group_cols: Optional[List[str]] = None,
    repeats: int = 5,
    embargo: int = 0,
    purge: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # helper: build GPU-accelerated tree when available (xgboost preferred)
    def _build_model(is_cls: bool):
        # try xgboost
        try:
            import xgboost as xgb  # type: ignore
            base = dict(
                n_estimators=int(n_estimators),
                max_depth=0,
                subsample=1.0,
                colsample_bytree=1.0,
                random_state=int(random_state),
                n_jobs=-1,
                verbosity=0,
            )
            try:
                params_gpu = dict(base)
                params_gpu.update(tree_method="hist", predictor="auto", device="cuda")
                return xgb.XGBClassifier(**params_gpu, eval_metric="logloss") if is_cls else xgb.XGBRegressor(**params_gpu, eval_metric="rmse")
            except TypeError:
                params_legacy = dict(base)
                params_legacy.update(tree_method="gpu_hist", predictor="gpu_predictor")
                return xgb.XGBClassifier(**params_legacy, eval_metric="logloss") if is_cls else xgb.XGBRegressor(**params_legacy, eval_metric="rmse")
        except Exception:
            pass
        # try catboost
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
            common_cb = dict(
                iterations=int(n_estimators),
                depth=6,
                random_seed=int(random_state),
                task_type="GPU",
                verbose=False,
            )
            return CatBoostClassifier(**common_cb) if is_cls else CatBoostRegressor(**common_cb)
        except Exception:
            pass
        # fallback sklearn RF (CPU)
        if is_cls:
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=random_state,
            )
        else:
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
            )
    if is_cls:
        clf = _build_model(True)
        y_tr_bin = y_tr.fillna(0).clip(0, 1).astype(int)
        y_va_bin = y_va.fillna(0).clip(0, 1).astype(int)
        clf.fit(X_tr, y_tr_bin)
        # prediction prob for both sklearn and xgboost/catboost
        try:
            prob = clf.predict_proba(X_va)[:, 1]
        except Exception:
            try:
                prob = clf.predict(X_va)
                if prob.ndim > 1:
                    prob = prob[:, 1]
            except Exception:
                prob = np.zeros(len(X_va), dtype=float)
        if time_perm:
            assert val_df is not None, "val_df required when time_perm is True"
            group_index = _build_group_index(val_df, group_cols)
            valid_mask = _build_valid_mask(len(X_va), group_index, embargo, purge)
        else:
            valid_mask = np.ones(len(X_va), dtype=bool)
        if valid_mask.any():
            base_score = f1_score(y_va_bin[valid_mask], (prob[valid_mask] >= 0.5).astype(int))
        else:
            base_score = float("nan")
        if not time_perm:
            perm = permutation_importance(
                clf,
                X_va,
                y_va_bin,
                scoring="f1",
                n_repeats=10,
                n_jobs=1,
                random_state=random_state,
            )
            try:
                importances = getattr(clf, "feature_importances_")
            except Exception:
                try:
                    importances = clf.get_feature_importance()
                except Exception:
                    importances = np.zeros(X_va.shape[1], dtype=float)
            return (
                importances,
                perm.importances_mean,
                perm.importances_std,
                float(base_score),
            )
        # time-aware permutation
        rng = np.random.RandomState(random_state)
        cols = list(X_va.columns)
        means, stds = [], []
        base = float(base_score)

        def _score(Xtmp: pd.DataFrame) -> float:
            prob_tmp = clf.predict_proba(Xtmp)[:, 1]
            if not valid_mask.any():
                return base
            return f1_score(y_va_bin[valid_mask], (prob_tmp[valid_mask] >= 0.5).astype(int))

        for c in cols:
            deltas = []
            for r in range(int(repeats)):
                Xp = X_va.copy()
                arr = Xp[c].to_numpy()
                for idx in group_index.values():
                    if len(idx) <= 1:
                        continue
                    max_shift = len(idx) - 1
                    if perm_method != "cyclic_shift":
                        raise NotImplementedError(f"perm_method={perm_method} not supported")
                    shift = block_len if block_len <= max_shift else max_shift
                    if shift <= 0:
                        shift = 1
                    # randomise direction/amount slightly to avoid deterministic artefacts
                    shift = min(max_shift, max(1, int(rng.normal(loc=shift, scale=max(1.0, shift * 0.1)))))
                    shift = max(1, min(max_shift, shift))
                    arr[idx] = np.roll(arr[idx], shift)
                Xp[c] = arr
                sc = _score(Xp)
                deltas.append(base - float(sc))
            means.append(float(np.mean(deltas)))
            stds.append(float(np.std(deltas)))
        return clf.feature_importances_, np.array(means), np.array(stds), float(base_score)
    # regression branch
    reg = _build_model(False)
    reg.fit(X_tr, y_tr)
    pred = reg.predict(X_va)
    if time_perm:
        assert val_df is not None, "val_df required when time_perm is True"
        group_index = _build_group_index(val_df, group_cols)
        valid_mask = _build_valid_mask(len(X_va), group_index, embargo, purge)
    else:
        valid_mask = np.ones(len(X_va), dtype=bool)
    if valid_mask.any():
        rmse = mean_squared_error(y_va[valid_mask], pred[valid_mask], squared=False)
    else:
        rmse = mean_squared_error(y_va, pred, squared=False)
    base_score = -float(rmse)
    if not time_perm:
        perm = permutation_importance(
            reg,
            X_va,
            y_va,
            scoring="neg_mean_squared_error",
            n_repeats=10,
            n_jobs=1,
            random_state=random_state,
        )
        try:
            importances = getattr(reg, "feature_importances_")
        except Exception:
            try:
                importances = reg.get_feature_importance()
            except Exception:
                importances = np.zeros(X_va.shape[1], dtype=float)
        return importances, perm.importances_mean, perm.importances_std, float(base_score)
    rng = np.random.RandomState(random_state)
    cols = list(X_va.columns)
    means, stds = [], []
    base = float(base_score)

    def _score_reg(Xtmp: pd.DataFrame) -> float:
        pred_tmp = reg.predict(Xtmp)
        if not valid_mask.any():
            return base
        rmse_tmp = mean_squared_error(y_va[valid_mask], pred_tmp[valid_mask], squared=False)
        return -float(rmse_tmp)

    for c in cols:
        deltas = []
        for r in range(int(repeats)):
            Xp = X_va.copy()
            arr = Xp[c].to_numpy()
            for idx in group_index.values():
                if len(idx) <= 1:
                    continue
                max_shift = len(idx) - 1
                if perm_method != "cyclic_shift":
                    raise NotImplementedError(f"perm_method={perm_method} not supported")
                shift = block_len if block_len <= max_shift else max_shift
                if shift <= 0:
                    shift = 1
                shift = min(max_shift, max(1, int(rng.normal(loc=shift, scale=max(1.0, shift * 0.1)))))
                shift = max(1, min(max_shift, shift))
                arr[idx] = np.roll(arr[idx], shift)
            Xp[c] = arr
            sc = _score_reg(Xp)
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
    allowlist_path: Optional[str] = None,
    pkl_path: Optional[str] = None,
    time_perm: bool = False,
    perm_method: str = "cyclic_shift",
    block_len: int = 36,
    block_len_by_period: Optional[Dict[str, int]] = None,
    group_cols: Optional[List[str]] = None,
    repeats: int = 5,
    targets_override: Optional[List[str]] = None,
    embargo: int = 0,
    embargo_by_period: Optional[Dict[str, int]] = None,
    purge: int = 0,
    purge_by_period: Optional[Dict[str, int]] = None,
    n_estimators: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    ds = load_split(
        pkl_path=pkl_path or "data/pkl_merged/full_merged.pkl",
        val_mode=val_mode,
        val_days=val_days,
        val_ratio=val_ratio,
        allowlist_path=allowlist_path,
        targets_override=targets_override,
    )
    os.makedirs(out_dir, exist_ok=True)
    periods = periods or ds.periods
    print(f"[tree_perm] start | periods={periods} targets={ds.targets} n_estimators={n_estimators}")
    summary_frames: List[pd.DataFrame] = []
    for per in tqdm(periods, desc="periods"):
        per_tr = ds.train[ds.train["period"].astype(str) == str(per)]
        per_va = ds.val[ds.val["period"].astype(str) == str(per)]
        if per_tr.empty or per_va.empty:
            print(f"[warn] period={per} has empty train/val; skip")
            continue
        per_block_len = int(block_len_by_period.get(str(per), block_len)) if isinstance(block_len_by_period, dict) else int(block_len)
        per_embargo = int(embargo_by_period.get(str(per), embargo)) if isinstance(embargo_by_period, dict) else int(embargo)
        per_purge = int(purge_by_period.get(str(per), purge)) if isinstance(purge_by_period, dict) else int(purge)
        per_dir = Path(out_dir) / str(per)
        per_dir.mkdir(parents=True, exist_ok=True)
        for target in tqdm(ds.targets, desc=f"targets@{per}"):
            if target not in per_tr.columns:
                continue
            is_cls = is_classification_target(target)
            print(f"[tree_perm] period={per} target={target} | fit+perm ...", end="", flush=True)
            X_tr, y_tr, X_va, y_va, _ = _prep_xy(per_tr, per_va, ds.features, target)
            importances, perm_delta, perm_std, base = fit_tree_and_permutation(
                X_tr,
                y_tr,
                X_va,
                y_va,
                is_cls=is_cls,
                n_estimators=n_estimators,
                random_state=random_state,
                time_perm=time_perm,
                val_df=per_va,
                perm_method=perm_method,
                block_len=per_block_len,
                group_cols=group_cols,
                repeats=repeats,
                embargo=per_embargo,
                purge=per_purge,
            )
            print(" done", flush=True)
            out = pd.DataFrame(
                {
                    "feature": X_tr.columns,
                    "tree_importance": importances,
                    "perm_mean": perm_delta,
                    "perm_std": perm_std,
                    "base_score": float(base),
                }
            )
            out["delta_metric"] = out["perm_mean"]
            out = out.sort_values(["tree_importance", "perm_mean"], ascending=[False, False]).reset_index(drop=True)
            n_feat = max(1, len(out))
            out["tree_rank"] = out["tree_importance"].rank(method="average", ascending=False)
            out["perm_rank"] = out["perm_mean"].rank(method="average", ascending=False)
            out["rank_avg"] = 0.5 * (out["tree_rank"] + out["perm_rank"])
            out["rank_pct"] = out["rank_avg"] / float(n_feat)
            out["period"] = str(per)
            out["target"] = target
            out["block_len"] = per_block_len
            out["embargo"] = per_embargo
            out["purge"] = per_purge
            csv_path = per_dir / f"{target}_importances.csv"
            out.to_csv(csv_path, index=False)
            print(f"[save] {csv_path} | base_score={base:.6f}")
            if topn_preview > 0:
                print(out.head(topn_preview)[["feature", "tree_importance", "perm_mean", "rank_pct"]].to_string(index=False))
            summary_frames.append(out)
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not summary.empty:
        summary_path = Path(out_dir) / "summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[tree_perm] done | {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Compute per-period & per-target feature importances (tree + permutation)")
    ap.add_argument("--periods", type=str, default=None, help="Comma-separated periods to include (default: all)")
    ap.add_argument("--val-mode", type=str, default="ratio", choices=["ratio", "days"])
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
    ap.add_argument("--embargo", type=int, default=0)
    ap.add_argument("--purge", type=int, default=0)
    ap.add_argument("--targets", type=str, default=None, help="Comma-separated targets to include (override)")
    args = ap.parse_args()
    targets_override = [t.strip() for t in (args.targets or "").split(",") if t.strip()] or None
    gcols = [c.strip() for c in (args.group_cols or "").split(",") if c.strip()]
    periods = None
    if args.periods:
        periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    summary = run(
        periods=periods or [],
        val_mode=args.val_mode,
        val_days=args.val_days,
        val_ratio=args.val_ratio,
        topn_preview=args.preview,
        out_dir=args.out,
        allowlist_path=args.allowlist,
        pkl_path=args.pkl,
        time_perm=bool(args.time_perm),
        perm_method="cyclic_shift",
        block_len=args.block_len,
        block_len_by_period=None,
        group_cols=gcols,
        repeats=args.perm_repeats,
        targets_override=targets_override,
        embargo=args.embargo,
        purge=args.purge,
    )
    if summary.empty:
        print("[warn] summary empty (no features evaluated)")


if __name__ == "__main__":
    main()
