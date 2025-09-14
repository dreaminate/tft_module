from __future__ import annotations
import argparse
import os
import random
from typing import List, Tuple, Dict

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


def _eval_subset(features: List[str], train_df: pd.DataFrame, val_df: pd.DataFrame, targets: List[str], weights_yaml: str, sample_cap: int | None = None) -> Tuple[float, Dict[str, float]]:
    import yaml
    with open(weights_yaml, "r", encoding="utf-8") as f:
        wcfg = yaml.safe_load(f)
    wlist = [float(x) for x in wcfg.get("custom_weights", [1.0] * len(targets))]
    wmap = {t: float(w) for t, w in zip(targets, wlist)}
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
    for t in targets:
        if t not in tr.columns:
            continue
        if is_classification_target(t):
            ytr = tr[t].fillna(0).clip(0, 1).astype(int); yva = va[t].fillna(0).clip(0, 1).astype(int)
            est = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced_subsample", random_state=42)
            est.fit(Xtr, ytr); prob = est.predict_proba(Xva)[:, 1]
            sc = f1_score(yva, (prob >= 0.5).astype(int))
        else:
            ytr = tr[t].astype(float); yva = va[t].astype(float)
            est = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
            est.fit(Xtr, ytr); pred = est.predict(Xva)
            try:
                rmse = float(mean_squared_error(yva, pred, squared=False))
            except TypeError:
                rmse = float(np.sqrt(mean_squared_error(yva, pred)))
            sc = -rmse
        score_by_target[t] = float(sc); w = float(wmap.get(t, 1.0)); total += w * float(sc); wsum += w
    if wsum <= 0:
        return -1e9, score_by_target
    return float(total / wsum), score_by_target


def rfe_search(pool: List[str], train_df: pd.DataFrame, val_df: pd.DataFrame, targets: List[str], weights_yaml: str, sample_cap: int | None = 50000, patience: int = 10) -> Tuple[List[str], float, Dict[str, float]]:
    keep = list(dict.fromkeys(pool))
    best_score, best_detail = _eval_subset(keep, train_df, val_df, targets, weights_yaml, sample_cap)
    improved, no_improve = True, 0
    print(f"[rfe] start with {len(keep)} features, score={best_score:.6f}")
    while improved and len(keep) > 1:
        improved = False
        cols = [c for c in keep if c in train_df.columns]
        Xtr = safe_numeric_copy(train_df[cols]).fillna(0).replace([np.inf, -np.inf], 0)
        ytr = train_df[targets[0]].fillna(0).astype(float)
        try:
            est = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
            est.fit(Xtr, ytr); importances = est.feature_importances_
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


def ga_search(pool: List[str], train_df: pd.DataFrame, val_df: pd.DataFrame, targets: List[str], weights_yaml: str, pop_size: int = 30, generations: int = 15, cx_prob: float = 0.7, mut_prob: float = 0.1, sample_cap: int | None = 50000) -> Tuple[List[str], float, Dict[str, float]]:
    rnd = random.Random(42); n = len(pool)
    def random_individual() -> List[int]:
        density = rnd.uniform(0.2, 0.5)
        return [1 if rnd.random() < density else 0 for _ in range(n)]
    def decode(ind: List[int]) -> List[str]:
        return [pool[i] for i, b in enumerate(ind) if b == 1]
    fitness_cache: Dict[tuple[int, ...], float] = {}
    def fitness(ind: List[int]) -> float:
        feats = decode(ind)
        if not feats:
            return -1e9
        sc, _ = _eval_subset(feats, train_df, val_df, targets, weights_yaml, sample_cap)
        pen = 0.0001 * sum(ind)
        return float(sc - pen)
    def get_fit(ind: List[int]) -> float:
        key = tuple(ind)
        if key not in fitness_cache:
            fitness_cache[key] = fitness(ind)
        return fitness_cache[key]
    def tournament(pop: List[List[int]], k: int = 3) -> List[int]:
        cand = rnd.sample(pop, k); cand.sort(key=lambda x: get_fit(x), reverse=True); return cand[0][:]
    def crossover(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        if rnd.random() > cx_prob or len(a) < 2:
            return a[:], b[:]
        p = rnd.randrange(1, len(a)); return a[:p] + b[p:], b[:p] + a[p:]
    def mutate(ind: List[int]) -> None:
        for i in range(len(ind)):
            if rnd.random() < mut_prob:
                ind[i] ^= 1
    pop = [random_individual() for _ in range(pop_size)]
    best_ind = max(pop, key=get_fit); best_fit = get_fit(best_ind)
    print(f"[ga] init best fitness={best_fit:.6f}, feats={sum(best_ind)}")
    for g in range(generations):
        new_pop: List[List[int]] = []
        while len(new_pop) < pop_size:
            p1 = tournament(pop); p2 = tournament(pop)
            c1, c2 = crossover(p1, p2); mutate(c1); mutate(c2); new_pop.extend([c1, c2])
        pop = new_pop[:pop_size]
        cur_best = max(pop, key=get_fit); cur_fit = get_fit(cur_best)
        if cur_fit > best_fit:
            best_ind, best_fit = cur_best, cur_fit
        print(f"[ga] gen {g+1}/{generations} best_fit={best_fit:.6f} feats={sum(best_ind)}")
    final_feats = decode(best_ind)
    final_score, details = _eval_subset(final_feats, train_df, val_df, targets, weights_yaml, sample_cap)
    return final_feats, final_score, details


def main():
    ap = argparse.ArgumentParser(description="Optimize feature subset by validation metric (GA/RFE)")
    ap.add_argument("--pool", type=str, default="configs/selected_features.txt")
    ap.add_argument("--core-csv", type=str, default="reports/feature_evidence/aggregated_core.csv")
    ap.add_argument("--method", type=str, default="rfe", choices=["rfe","ga"])
    ap.add_argument("--val-mode", type=str, default="ratio", choices=["ratio","days"]) 
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--sample-cap", type=int, default=50000)
    ap.add_argument("--weights", type=str, default="configs/weights_config.yaml")
    ap.add_argument("--out", type=str, default="reports/feature_evidence/optimized_features.txt")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--gen", type=int, default=15)
    ap.add_argument("--cx", type=float, default=0.7)
    ap.add_argument("--mut", type=float, default=0.1)
    ap.add_argument("--pkl", type=str, default=None, help="Override merged dataset PKL path")
    args = ap.parse_args()
    pool = _prep_pool(args.pool, args.core_csv)
    ds = load_split(pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl", val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio)
    if args.method == "rfe":
        feats, score, det = rfe_search(pool, ds.train, ds.val, ds.targets, args.weights, args.sample_cap)
    else:
        feats, score, det = ga_search(pool, ds.train, ds.val, ds.targets, args.weights, pop_size=args.pop, generations=args.gen, cx_prob=args.cx, mut_prob=args.mut, sample_cap=args.sample_cap)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in feats:
            f.write(c + "\n")
    print(f"[save] {args.out} ({len(feats)} features) | score={score:.6f}")
    print("[detail] per-target:")
    for k, v in det.items():
        print(f"  - {k}: {v:.6f}")


if __name__ == "__main__":
    main()
