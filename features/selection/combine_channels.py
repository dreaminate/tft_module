from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def _rank01(series: pd.Series, ascending: bool = False) -> pd.Series:
    return series.rank(method="average", ascending=ascending, na_option="keep").astype(float) / max(1.0, float(series.count()))


def _load_pair_scores(dir_path: str) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for per in sorted(os.listdir(dir_path)):
        pdir = Path(dir_path) / per
        if not pdir.is_dir():
            continue
        for csv in pdir.glob("*_importances.csv"):
            target = csv.stem.replace("_importances", "")
            df = pd.read_csv(csv)
            df["score_local"] = 0.5 * _rank01(df["tree_importance"], ascending=False) + 0.5 * _rank01(df["perm_mean"], ascending=False)
            # base_score is duplicated per row; keep a single value
            base_score = float(df.get("base_score", pd.Series([np.nan])).iloc[0])
            df["base_score"] = base_score
            out[(str(per), str(target))] = df[["feature", "score_local", "base_score"]].copy()
    return out


def _score_to01(base_score: float) -> float:
    if not np.isfinite(base_score):
        return 0.5
    # classification F1 in [0,1] likely
    if base_score >= 0:
        return float(np.clip(base_score, 0.0, 1.0))
    # regression uses -RMSE; convert to 1/(1+RMSE)
    rmse = -float(base_score)
    return float(1.0 / (1.0 + max(0.0, rmse)))


def _load_weights(yaml_path: str) -> Dict[str, float]:
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ws = cfg.get("custom_weights", [])
    return {t: float(w) for t, w in zip(sorted(cfg.get("target_names", [])) or [], ws)}


def combine_channels(
    base_dir: str,
    rich_dir: str,
    weights_yaml: str,
    topk: int,
    topk_per_pair: int | None,
    min_appear_rate: float | None,
) -> pd.DataFrame:
    base_pairs = _load_pair_scores(base_dir) if base_dir and os.path.exists(base_dir) else {}
    rich_pairs = _load_pair_scores(rich_dir) if rich_dir and os.path.exists(rich_dir) else {}
    pairs = sorted(set(base_pairs.keys()) | set(rich_pairs.keys()))
    rows: List[pd.DataFrame] = []
    for (per, tgt) in pairs:
        db = base_pairs.get((per, tgt))
        dr = rich_pairs.get((per, tgt))
        sb = _score_to01(float(db["base_score"].iloc[0])) if db is not None else 0.0
        sr = _score_to01(float(dr["base_score"].iloc[0])) if dr is not None else 0.0
        denom = max(1e-8, sb + sr)
        w_r = float(sr / denom) if denom > 0 else 0.5
        w_b = 1.0 - w_r
        # align features
        feats = set()
        if db is not None:
            feats.update(db["feature"].astype(str).tolist())
        if dr is not None:
            feats.update(dr["feature"].astype(str).tolist())
        if not feats:
            continue
        m: Dict[str, float] = {}
        if db is not None:
            m.update({str(r["feature"]): float(r["score_local"]) * w_b for _, r in db.iterrows()})
        if dr is not None:
            for _, r in dr.iterrows():
                f = str(r["feature"])
                m[f] = m.get(f, 0.0) + float(r["score_local"]) * w_r
        comb = pd.DataFrame({"feature": list(m.keys()), "score_pair": list(m.values())})
        comb["period"], comb["target"], comb["w_rich"], comb["w_base"] = per, tgt, w_r, w_b
        rows.append(comb)
    if not rows:
        raise SystemExit("[err] no pairs found in base/rich dirs")
    dfp = pd.concat(rows, ignore_index=True)
    # optional appearance rate across pairs (topk per pair)
    appear_rate = None
    if topk_per_pair and topk_per_pair > 0:
        marks = []
        for (per, tgt), g in dfp.groupby(["period", "target"], sort=False):
            gg = g.sort_values("score_pair", ascending=False).head(int(topk_per_pair))
            marks.extend([(f, per, tgt) for f in gg["feature"].astype(str).tolist()])
        mdf = pd.DataFrame(marks, columns=["feature", "period", "target"]).drop_duplicates()
        pair_count = max(1, int(mdf.groupby(["period", "target"]).ngroups))
        ar = mdf.groupby("feature").size().rename("appear_count").reset_index()
        ar["appear_rate"] = ar["appear_count"].astype(float) / float(pair_count)
        appear_rate = ar[["feature", "appear_rate"]]

    # aggregate across periods for each target then across targets with weights
    ft = dfp.groupby(["feature", "target"], as_index=False)["score_pair"].mean().rename(columns={"score_pair": "score_tp"})
    # weights: fallback to equal weights when missing
    # try to take from weights_yaml.custom_weights matching sorted unique targets
    target_list = sorted(ft["target"].unique().tolist())
    w_map = {t: 1.0 for t in target_list}
    if weights_yaml:
        try:
            with open(weights_yaml, "r", encoding="utf-8") as f:
                wc = yaml.safe_load(f) or {}
            ws = [float(x) for x in (wc.get("custom_weights") or [])]
            if len(ws) == len(target_list):
                w_map = {t: float(w) for t, w in zip(target_list, ws)}
        except Exception:
            pass
    ft["w"] = ft["target"].map(w_map).astype(float)
    g = ft.groupby("feature", as_index=False).apply(lambda gdf: pd.Series({
        "score_global": float((gdf["score_tp"] * gdf["w"]).sum() / max(1e-8, gdf["w"].sum()))
    })).reset_index()
    if appear_rate is not None and not appear_rate.empty:
        g = g.merge(appear_rate, on="feature", how="left")
        g["appear_rate"] = g["appear_rate"].fillna(0.0)
        if isinstance(min_appear_rate, (int, float)) and 0 <= float(min_appear_rate) <= 1:
            g = g[g["appear_rate"] >= float(min_appear_rate)].copy()
    else:
        g["appear_rate"] = np.nan
    g = g.sort_values("score_global", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    g["keep"] = g["rank"] <= int(topk)
    return g


def main():
    ap = argparse.ArgumentParser(description="Combine base & rich channel evidences via validation-gated fusion")
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--rich", type=str, required=True)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--topk-per-pair", type=int, default=64)
    ap.add_argument("--min-appear-rate", type=float, default=0.5)
    ap.add_argument("--out-summary", type=str, required=True)
    ap.add_argument("--out-allowlist", type=str, required=True)
    args = ap.parse_args()
    core = combine_channels(
        base_dir=args.base,
        rich_dir=args.rich,
        weights_yaml=args.weights,
        topk=int(args.topk),
        topk_per_pair=int(args.topk_per_pair),
        min_appear_rate=float(args.min_appear_rate),
    )
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    core.to_csv(args.out_summary, index=False)
    keep = core[core["keep"]]["feature"].astype(str).tolist()
    os.makedirs(os.path.dirname(args.out_allowlist), exist_ok=True)
    with open(args.out_allowlist, "w", encoding="utf-8") as f:
        for c in keep:
            f.write(c + "\n")
    print(f"[save] {args.out_summary}")
    print(f"[save] {args.out_allowlist} ({len(keep)} features)")


if __name__ == "__main__":
    main()
