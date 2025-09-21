from __future__ import annotations
import argparse
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error

from .common import load_split, is_classification_target, safe_numeric_copy


def _evaluate(
    features: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    targets: List[str],
    weights_yaml: str | None,
) -> float:
    import yaml
    ws = [1.0] * len(targets)
    if weights_yaml and os.path.exists(weights_yaml):
        try:
            with open(weights_yaml, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            ws = [float(x) for x in (cfg.get("custom_weights") or ws)]
        except Exception:
            pass
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
            # prefer GPU XGBoost, then CatBoost, fallback sklearn RF
            est = None
            try:
                import xgboost as xgb  # type: ignore
                try:
                    est = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=0,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        random_state=1,
                        n_jobs=-1,
                        tree_method="hist",
                        predictor="auto",
                        device="cuda",
                        verbosity=0,
                        eval_metric="logloss",
                    )
                except TypeError:
                    est = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=0,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        random_state=1,
                        n_jobs=-1,
                        tree_method="gpu_hist",
                        predictor="gpu_predictor",
                        verbosity=0,
                        eval_metric="logloss",
                    )
            except Exception:
                try:
                    from catboost import CatBoostClassifier  # type: ignore
                    est = CatBoostClassifier(
                        iterations=200,
                        depth=6,
                        random_seed=1,
                        task_type="GPU",
                        verbose=False,
                    )
                except Exception:
                    est = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced_subsample", random_state=1)
            est.fit(Xtr, ytr)
            try:
                prob = est.predict_proba(Xva)[:, 1]
            except Exception:
                pred = est.predict(Xva)
                prob = pred if getattr(pred, "ndim", 1) == 1 else pred[:, 1]
            sc = f1_score(yva, (prob >= 0.5).astype(int))
        else:
            ytr = train_df[t].astype(float)
            yva = val_df[t].astype(float)
            try:
                import xgboost as xgb  # type: ignore
                try:
                    est = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=0,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        random_state=1,
                        n_jobs=-1,
                        tree_method="hist",
                        predictor="auto",
                        device="cuda",
                        verbosity=0,
                        eval_metric="rmse",
                    )
                except TypeError:
                    est = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=0,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        random_state=1,
                        n_jobs=-1,
                        tree_method="gpu_hist",
                        predictor="gpu_predictor",
                        verbosity=0,
                        eval_metric="rmse",
                    )
            except Exception:
                try:
                    from catboost import CatBoostRegressor  # type: ignore
                    est = CatBoostRegressor(
                        iterations=200,
                        depth=6,
                        random_seed=1,
                        task_type="GPU",
                        verbose=False,
                    )
                except Exception:
                    est = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=1)
            est.fit(Xtr, ytr); pred = est.predict(Xva)
            try:
                sc = -mean_squared_error(yva, pred, squared=False)
            except TypeError:
                sc = -float(np.sqrt(mean_squared_error(yva, pred)))
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


def evaluate_post_validation(
    features: List[str],
    dataset_split,
    config: Dict[str, object],
    weights_yaml: str | None,
    periods: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, object]:
    method = str(config.get("method", "rolling")).lower()
    if method not in {"rolling"}:
        raise ValueError(f"unsupported post_validation method={method}")
    n_splits = int(config.get("n_splits", 5))
    val_ratio = float(config.get("val_ratio", 0.2))
    periods = periods or dataset_split.periods
    results: List[Dict[str, float]] = []
    for per in periods:
        dfp = dataset_split.train[dataset_split.train["period"].astype(str) == str(per)].copy()
        if dfp.empty:
            continue
        splits = rolling_splits_period(dfp, n_splits=n_splits, val_ratio=val_ratio)
        if not splits:
            results.append({"period": per, "folds": 0, "mean_score": float("nan")})
            continue
        scores = []
        for tr, va in splits:
            sc = _evaluate(features, tr, va, dataset_split.targets, weights_yaml)
            scores.append(sc)
        mean_sc = float(np.mean(scores)) if scores else float("nan")
        results.append({"period": per, "folds": len(scores), "mean_score": mean_sc})
    summary: Dict[str, object] = {
        "method": method,
        "results": results,
    }
    if results:
        valid_scores = [r["mean_score"] for r in results if isinstance(r["mean_score"], float) and not np.isnan(r["mean_score"])]
        if valid_scores:
            summary["mean_score"] = float(np.mean(valid_scores))
            summary["min_score"] = float(np.min(valid_scores))
    if out_dir and results:
        out_path = Path(out_dir) / "post_validation.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        summary["report_path"] = str(out_path)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Rolling time-series validation for a selected feature set")
    ap.add_argument("--features", type=str, default="configs/selected_features.txt")
    ap.add_argument("--val-mode", type=str, default="ratio", choices=["ratio","days"]) 
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--periods", type=str, default=None)
    ap.add_argument("--pkl", type=str, default=None, help="Override merged dataset PKL path")
    args = ap.parse_args()
    with open(args.features, "r", encoding="utf-8") as f:
        feats = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    ds = load_split(pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl", val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio)
    periods = args.periods.split(",") if args.periods else ds.periods
    cfg = {"method": "rolling", "n_splits": args.splits, "val_ratio": args.val_ratio}
    summary = evaluate_post_validation(feats, ds, cfg, args.weights, periods=periods, out_dir="reports/feature_evidence")
    if summary.get("results"):
        print("[post] summary:")
        for item in summary["results"]:
            print(f"  period={item['period']} folds={item['folds']} mean_score={item['mean_score']}")
    else:
        print("[warn] no rolling results produced")


if __name__ == "__main__":
    main()
