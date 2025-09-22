from __future__ import annotations
import argparse
import os
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error

from .common import load_split, is_classification_target, safe_numeric_copy


def _prep_pool(pool_file: str | None, core_csv: str | None, default_topk: int = 128) -> List[str]:
    if pool_file and os.path.exists(pool_file):
        with open(pool_file, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if core_csv and os.path.exists(core_csv):
        df = pd.read_csv(core_csv)
        if "feature" in df.columns:
            return df.head(default_topk)["feature"].astype(str).tolist()
    raise SystemExit("[err] no candidate feature list provided (pool_file/core_csv missing)")


def _eval_subset(
    features: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    targets: List[str],
    weights_yaml: str | None,
    sample_cap: int | None = None,
) -> Tuple[float, Dict[str, float]]:
    import yaml
    ws = [1.0] * len(targets)
    if weights_yaml and os.path.exists(weights_yaml):
        try:
            with open(weights_yaml, "r", encoding="utf-8") as f:
                wcfg = yaml.safe_load(f) or {}
            ws = [float(x) for x in (wcfg.get("custom_weights") or ws)]
        except Exception:
            pass
    wmap = {t: float(w) for t, w in zip(targets, ws)}
    tr = train_df.copy(); va = val_df.copy(); cols = [c for c in features if c in tr.columns]
    if not cols:
        return -1e9, {}
    Xtr = safe_numeric_copy(tr[cols]); Xva = safe_numeric_copy(va[cols])
    med = Xtr.median(axis=0, numeric_only=True)
    Xtr = Xtr.fillna(med).replace([np.inf, -np.inf], 0)
    Xva = Xva.fillna(med).replace([np.inf, -np.inf], 0)
    if sample_cap and len(Xtr) > sample_cap:
        idx = np.random.RandomState(123).choice(len(Xtr), size=sample_cap, replace=False)
        Xtr = Xtr.iloc[idx]; tr = tr.iloc[idx]
    score_by_target: Dict[str, float] = {}; total, wsum = 0.0, 0.0
    cls_scores: List[float] = []
    reg_scores: List[float] = []
    for t in targets:
        if t not in tr.columns:
            continue
        if is_classification_target(t):
            ytr = tr[t].fillna(0).clip(0, 1).astype(int); yva = va[t].fillna(0).clip(0, 1).astype(int)
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
                        random_state=42,
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
                        random_state=42,
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
                        random_seed=42,
                        task_type="GPU",
                        verbose=False,
                    )
                except Exception:
                    est = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced_subsample", random_state=42)
            est.fit(Xtr, ytr)
            try:
                prob = est.predict_proba(Xva)[:, 1]
            except Exception:
                pred = est.predict(Xva)
                prob = pred if pred.ndim == 1 else pred[:, 1]
            sc = f1_score(yva, (prob >= 0.5).astype(int))
            cls_scores.append(sc)
        else:
            ytr = tr[t].astype(float); yva = va[t].astype(float)
            try:
                import xgboost as xgb  # type: ignore
                try:
                    est = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=0,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        random_state=42,
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
                        random_state=42,
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
                        random_seed=42,
                        task_type="GPU",
                        verbose=False,
                    )
                except Exception:
                    est = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
            est.fit(Xtr, ytr); pred = est.predict(Xva)
            try:
                from sklearn.metrics import root_mean_squared_error as _rmse  # type: ignore
                rmse = float(_rmse(yva, pred))
            except Exception:
                rmse = float(np.sqrt(mean_squared_error(yva, pred)))
            sc = -rmse
            reg_scores.append(sc)
        score_by_target[t] = float(sc); w = float(wmap.get(t, 1.0)); total += w * float(sc); wsum += w
    if wsum <= 0:
        overall = -1e9
    else:
        overall = float(total / wsum)
    detail = {
        "score_by_target": score_by_target,
        "classification_avg": float(np.mean(cls_scores)) if cls_scores else None,
        "regression_avg": float(np.mean(reg_scores)) if reg_scores else None,
    }
    return overall, detail


def rfe_search(
    pool: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    targets: List[str],
    weights_yaml: str,
    sample_cap: int | None = 50000,
    patience: int = 10,
) -> Tuple[List[str], float, Dict[str, float]]:
    keep = list(dict.fromkeys(pool))
    best_score, best_detail = _eval_subset(keep, train_df, val_df, targets, weights_yaml, sample_cap)
    improved, no_improve = True, 0
    print(f"[rfe] start with {len(keep)} features, score={best_score:.6f}")
    while improved and len(keep) > 1:
        improved = False
        cols = [c for c in keep if c in train_df.columns]
        Xtr = safe_numeric_copy(train_df[cols]).fillna(0).replace([np.inf, -np.inf], 0)
        ytr = train_df[targets[0]].fillna(0).astype(float)
        # 近似器优先选择 LightGBM，其次 CatBoost，再次 XGBoost，最后 RF
        try:
            try:
                import lightgbm as lgb  # type: ignore
                est = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1)
                est.fit(Xtr, ytr)
                importances = getattr(est, "feature_importances_", None)
                if importances is None:
                    raise ValueError("no importances")
            except Exception:
                try:
                    from catboost import CatBoostRegressor  # type: ignore
                    est = CatBoostRegressor(iterations=300, depth=6, random_seed=0, task_type="GPU", verbose=False)
                    est.fit(Xtr, ytr)
                    importances = est.get_feature_importance()
                except Exception:
                    try:
                        import xgboost as xgb  # type: ignore
                        est = xgb.XGBRegressor(n_estimators=300, max_depth=0, subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=-1, tree_method="hist", predictor="auto", device="cuda", verbosity=0, eval_metric="rmse")
                        est.fit(Xtr, ytr)
                        importances = getattr(est, "feature_importances_", None)
                        if importances is None:
                            raise ValueError("no importances")
                    except Exception:
                        from sklearn.ensemble import RandomForestRegressor as _RF
                        est = _RF(n_estimators=300, n_jobs=-1, random_state=0)
                        est.fit(Xtr, ytr)
                        importances = getattr(est, "feature_importances_", np.ones(len(cols)) / len(cols))
        except Exception:
            importances = np.ones(len(cols)) / len(cols)
        order = np.argsort(importances)
        step = max(1, len(keep) // 20); candidates = [cols[i] for i in order[:step]]
        trial_best_score, trial_best_keep, trial_best_detail = best_score, keep, best_detail
        for c in candidates:
            new_keep = [f for f in keep if f != c]
            sc, det = _eval_subset(new_keep, train_df, val_df, targets, weights_yaml, sample_cap)
            if sc > trial_best_score + 1e-6:
                trial_best_score, trial_best_keep, trial_best_detail = sc, new_keep, det
        if trial_best_score > best_score + 1e-6:
            keep, best_score, best_detail = trial_best_keep, trial_best_score, trial_best_detail
            improved, no_improve = True, 0
            print(f"[rfe] improved -> {len(keep)} features, score={best_score:.6f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return keep, best_score, best_detail


def ga_search(
    pool: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    targets: List[str],
    weights_yaml: str,
    pop_size: int = 30,
    generations: int = 15,
    cx_prob: float = 0.7,
    mut_prob: float = 0.1,
    sample_cap: int | None = 50000,
    multi_objective: bool = False,
    mo_weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Tuple[List[str], float, Dict[str, float]]:
    rnd = random.Random(int(seed)); n = len(pool)
    mo_weights = mo_weights or {"classification": 1.0, "regression": 1.0}

    def random_individual() -> List[int]:
        density = rnd.uniform(0.2, 0.5)
        return [1 if rnd.random() < density else 0 for _ in range(n)]

    def decode(ind: List[int]) -> List[str]:
        return [pool[i] for i, b in enumerate(ind) if b == 1]

    fitness_cache: Dict[tuple[int, ...], Tuple[float, Dict[str, float]]] = {}

    def fitness(ind: List[int]) -> Tuple[float, Dict[str, float]]:
        feats = decode(ind)
        if not feats:
            return -1e9, {}
        sc, detail = _eval_subset(feats, train_df, val_df, targets, weights_yaml, sample_cap)
        if multi_objective:
            cls_avg = detail.get("classification_avg")
            reg_avg = detail.get("regression_avg")
            agg = 0.0
            used = 0.0
            if cls_avg is not None:
                agg += float(mo_weights.get("classification", 1.0)) * cls_avg
                used += abs(float(mo_weights.get("classification", 1.0)))
            if reg_avg is not None:
                agg += float(mo_weights.get("regression", 1.0)) * reg_avg
                used += abs(float(mo_weights.get("regression", 1.0)))
            if used > 0:
                sc = agg / used
            detail = dict(detail)
            detail["aggregate_score"] = sc
        return sc, detail

    def get_fit(ind: List[int]) -> Tuple[float, Dict[str, float]]:
        key = tuple(ind)
        if key not in fitness_cache:
            fitness_cache[key] = fitness(ind)
        return fitness_cache[key]

    def tournament(pop: List[List[int]], k: int = 3) -> List[int]:
        cand = rnd.sample(pop, k)
        cand.sort(key=lambda x: get_fit(x)[0], reverse=True)
        return cand[0][:]

    def crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        if rnd.random() > cx_prob or len(a) < 2:
            return a[:], b[:]
        p = rnd.randrange(1, len(a))
        return a[:p] + b[p:], b[:p] + a[p:]

    def mutate(ind: List[int]) -> None:
        for i in range(len(ind)):
            if rnd.random() < mut_prob:
                ind[i] ^= 1

    pop = [random_individual() for _ in range(pop_size)]
    best_ind = max(pop, key=lambda x: get_fit(x)[0]); best_fit = get_fit(best_ind)[0]
    print(f"[ga] init best fitness={best_fit:.6f}, feats={sum(best_ind)}")
    for g in range(generations):
        new_pop: List[List[int]] = []
        while len(new_pop) < pop_size:
            p1 = tournament(pop); p2 = tournament(pop)
            c1, c2 = crossover(p1, p2); mutate(c1); mutate(c2); new_pop.extend([c1, c2])
        pop = new_pop[:pop_size]
        cur_best = max(pop, key=lambda x: get_fit(x)[0]); cur_fit = get_fit(cur_best)[0]
        if cur_fit > best_fit:
            best_ind, best_fit = cur_best, cur_fit
        print(f"[ga] gen {g+1}/{generations} best_fit={best_fit:.6f} feats={sum(best_ind)}")
    final_feats = decode(best_ind)
    final_score, details = fitness(best_ind)
    return final_feats, final_score, details


def ga_search_multi(
    pool: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    targets: List[str],
    weights_yaml: str,
    seeds: List[int],
    pop_size: int = 30,
    generations: int = 15,
    cx_prob: float = 0.7,
    mut_prob: float = 0.1,
    sample_cap: int | None = 50000,
    multi_objective: bool = False,
    mo_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], float, Dict[str, float], Dict[str, float]]:
    best_feats: List[str] = []
    best_score: float = -1e9
    best_detail: Dict[str, float] = {}
    locus_freq: Dict[str, int] = {}
    for sd in seeds:
        feats, score, det = ga_search(
            pool,
            train_df,
            val_df,
            targets,
            weights_yaml,
            pop_size=pop_size,
            generations=generations,
            cx_prob=cx_prob,
            mut_prob=mut_prob,
            sample_cap=sample_cap,
            multi_objective=multi_objective,
            mo_weights=mo_weights,
            seed=int(sd),
        )
        for f in feats:
            locus_freq[f] = locus_freq.get(f, 0) + 1
        if score > best_score:
            best_feats, best_score, best_detail = feats, score, det
    total = float(max(1, len(seeds)))
    freq_pct = {k: float(v) / total for k, v in locus_freq.items()}
    return best_feats, best_score, best_detail, freq_pct


def main():
    ap = argparse.ArgumentParser(description="Optimize feature subset by validation metric (GA/RFE)")
    ap.add_argument("--pool", type=str, default="configs/selected_features.txt")
    ap.add_argument("--core-csv", type=str, default="reports/feature_evidence/aggregated_core.csv")
    ap.add_argument("--method", type=str, default="rfe", choices=["rfe","ga"])
    ap.add_argument("--val-mode", type=str, default="ratio", choices=["ratio","days"]) 
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--sample-cap", type=int, default=50000)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--out", type=str, default="reports/feature_evidence/optimized_features.txt")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gen", type=int, default=15)
    ap.add_argument("--cx", type=float, default=0.7)
    ap.add_argument("--mut", type=float, default=0.1)
    ap.add_argument("--multi-objective", action="store_true")
    ap.add_argument("--mo-weight-cls", type=float, default=1.0)
    ap.add_argument("--mo-weight-reg", type=float, default=1.0)
    ap.add_argument("--pkl", type=str, default=None, help="Override merged dataset PKL path")
    args = ap.parse_args()
    pool = _prep_pool(args.pool, args.core_csv)
    ds = load_split(pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl", val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio)
    if args.method == "rfe":
        feats, score, det = rfe_search(pool, ds.train, ds.val, ds.targets, args.weights, args.sample_cap)
    else:
        feats, score, det = ga_search(
            pool,
            ds.train,
            ds.val,
            ds.targets,
            args.weights,
            pop_size=args.pop,
            generations=args.gen,
            cx_prob=args.cx,
            mut_prob=args.mut,
            sample_cap=args.sample_cap,
            multi_objective=bool(args.multi_objective),
            mo_weights={"classification": args.mo_weight_cls, "regression": args.mo_weight_reg},
        )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in feats:
            f.write(c + "\n")
    print(f"[save] {args.out} ({len(feats)} features) | score={score:.6f}")
    print("[detail] per-target:")
    by_target = det.get("score_by_target", {}) if isinstance(det, dict) else {}
    for k, v in by_target.items():
        print(f"  - {k}: {v:.6f}")
    if isinstance(det, dict):
        cls_avg = det.get("classification_avg")
        reg_avg = det.get("regression_avg")
        if cls_avg is not None:
            print(f"  * classification_avg: {cls_avg:.6f}")
        if reg_avg is not None:
            print(f"  * regression_avg: {reg_avg:.6f}")


if __name__ == "__main__":
    main()
