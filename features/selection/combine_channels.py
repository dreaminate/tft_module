from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from .aggregate_core import aggregate_tree_perm, build_unified_core


def _score_to01(base_score: float) -> float:
    if not np.isfinite(base_score):
        return 0.5
    if base_score >= 0:
        return float(np.clip(base_score, 0.0, 1.0))
    rmse = -float(base_score)
    return float(1.0 / (1.0 + max(0.0, rmse)))


def _group_pairs(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    if df.empty:
        return {}
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (per, tgt), g in df.groupby(["period", "target"], sort=False):
        out[(str(per), str(tgt))] = g
    return out


def _fuse_pair_scores_with_quality(
    base_df: pd.DataFrame,
    rich_df: pd.DataFrame,
    rich_quality_weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    rich_quality_weights = rich_quality_weights or {}
    base_pairs = _group_pairs(base_df)
    rich_pairs = _group_pairs(rich_df)
    all_pairs = sorted(set(base_pairs.keys()) | set(rich_pairs.keys()))
    rows: List[Dict[str, float]] = []
    for per, tgt in all_pairs:
        db = base_pairs.get((per, tgt))
        dr = rich_pairs.get((per, tgt))
        sb = _score_to01(float(db["base_score"].dropna().iloc[0])) if db is not None and "base_score" in db.columns else 0.0
        sr = _score_to01(float(dr["base_score"].dropna().iloc[0])) if dr is not None and "base_score" in dr.columns else 0.0
        denom = max(1e-8, sb + sr)
        w_r = float(sr / denom) if denom > 0 else 0.5
        # quality weight per period (e.g., based on missing rate/latency)
        q = float(rich_quality_weights.get(str(per), 1.0))
        w_r = max(0.0, min(1.0, w_r * q))
        w_b = 1.0 - w_r
        fused: Dict[str, float] = {}
        base_vals: Dict[str, float] = {}
        rich_vals: Dict[str, float] = {}
        # propagate era if available
        era_tag = None
        if db is not None and "era" in db.columns and not db["era"].isna().all():
            era_tag = str(db["era"].dropna().iloc[0])
        if era_tag is None and dr is not None and "era" in dr.columns and not dr["era"].isna().all():
            era_tag = str(dr["era"].dropna().iloc[0])
        if db is not None:
            for _, r in db.iterrows():
                feat = str(r["feature"])
                val = float(r.get("score_local", r.get("score_tree", 0.0)))
                fused[feat] = fused.get(feat, 0.0) + w_b * val
                base_vals[feat] = val
        if dr is not None:
            for _, r in dr.iterrows():
                feat = str(r["feature"])
                val = float(r.get("score_local", r.get("score_tree", 0.0)))
                fused[feat] = fused.get(feat, 0.0) + w_r * val
                rich_vals[feat] = val
        if not fused:
            continue
        for feat, score in fused.items():
            row = {
                "feature": feat,
                "period": per,
                "target": tgt,
                "score_local": score,
                "score_base": base_vals.get(feat, 0.0),
                "score_rich": rich_vals.get(feat, 0.0),
                "w_base": w_b,
                "w_rich": w_r,
                "base_metric": sb,
                "rich_metric": sr,
            }
            if era_tag is not None:
                row["era"] = era_tag
            rows.append(row)
    if not rows:
        raise SystemExit("[err] no overlapping pairs to combine for base/rich")
    dfp = pd.DataFrame(rows)
    grp = dfp.groupby(["period", "target"], sort=False)
    counts = grp["feature"].transform("count").astype(float).clip(lower=1.0)
    dfp["rank_pct"] = grp["score_local"].rank(method="average", ascending=False) / counts
    return dfp


def combine_channels(
    base_dir: str,
    rich_dir: str,
    weights_yaml: str,
    topk: int,
    topk_per_pair: int | None,
    min_appear_rate: float | None,
    rich_quality_weights: Dict[str, float] | None = None,
    min_appear_rate_era: float | None = None,
) -> pd.DataFrame:
    base_df = aggregate_tree_perm(base_dir) if base_dir and os.path.exists(base_dir) else pd.DataFrame()
    rich_df = aggregate_tree_perm(rich_dir) if rich_dir and os.path.exists(rich_dir) else pd.DataFrame()
    fused_pairs = _fuse_pair_scores_with_quality(base_df, rich_df, rich_quality_weights)
    core = build_unified_core(
        fused_pairs,
        weights_yaml,
        topk=topk,
        tft_bonus=0.0,
        tft_file=None,
        topk_per_pair=topk_per_pair,
        min_appear_rate=min_appear_rate,
        min_appear_rate_era=min_appear_rate_era,
    )
    return core


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
            f.write(f"{c}\n")
    print(f"[save] {args.out_summary}")
    print(f"[save] {args.out_allowlist} ({len(keep)} features)")


if __name__ == "__main__":
    main()
