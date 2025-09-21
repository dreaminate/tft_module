from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def _load_weights(yaml_path: str | None) -> List[float]:
    if not yaml_path or not os.path.exists(yaml_path):
        return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return [float(w) for w in cfg.get("custom_weights", [])]


def _read_summary(in_dir: Path) -> pd.DataFrame | None:
    summary_path = in_dir / "summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return None


def _fallback_collect(in_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for per in sorted(p for p in os.listdir(in_dir) if (in_dir / p).is_dir()):
        per_dir = in_dir / per
        for csv in per_dir.glob("*_importances.csv"):
            target = csv.stem.replace("_importances", "")
            df = pd.read_csv(csv)
            df["period"] = str(per)
            df["target"] = target
            rows.append(df)
    if not rows:
        raise SystemExit(f"[err] no importances found under {in_dir}")
    return pd.concat(rows, ignore_index=True)


def _ensure_pair_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "delta_metric" not in work.columns:
        if "perm_mean" in work.columns:
            work["delta_metric"] = work["perm_mean"]
        else:
            raise ValueError("missing delta_metric/perm_mean in permutation summary")
    if "base_score" not in work.columns:
        work["base_score"] = np.nan
    required = {"feature", "period", "target", "tree_importance", "delta_metric"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"missing columns in permutation summary: {missing}")
    groups = work.groupby(["period", "target"], sort=False)
    work["tree_rank"] = groups["tree_importance"].rank(method="average", ascending=False)
    work["perm_rank"] = groups["delta_metric"].rank(method="average", ascending=False)
    counts = groups["feature"].transform("count").astype(float).clip(lower=1.0)
    denom = np.where(counts > 1.0, counts - 1.0, 1.0)
    work["score_tree"] = 1.0 - (work["tree_rank"] - 1.0) / denom
    work["score_perm"] = 1.0 - (work["perm_rank"] - 1.0) / denom
    work["score_tree"] = work["score_tree"].fillna(0.0)
    work["score_perm"] = work["score_perm"].fillna(0.0)
    work["score_local"] = 0.5 * work["score_tree"] + 0.5 * work["score_perm"]
    work["rank_avg"] = 0.5 * (work["tree_rank"] + work["perm_rank"])
    work["rank_pct"] = work["rank_avg"] / counts
    return work


def aggregate_tree_perm(in_dir: str) -> pd.DataFrame:
    root = Path(in_dir)
    summary = _read_summary(root)
    if summary is None:
        summary = _fallback_collect(root)
    return _ensure_pair_metrics(summary)


def build_unified_core(
    pair_df: pd.DataFrame,
    weights_yaml: str,
    topk: int = 128,
    tft_bonus: float = 0.0,
    tft_file: str | None = None,
    topk_per_pair: int | None = None,
    min_appear_rate: float | None = None,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(columns=["feature", "score_global", "appear_rate", "rank", "keep"])
    df = pair_df.copy()
    df["score_local"] = df["score_local"].fillna(0.0)

    marks = []
    if topk_per_pair and topk_per_pair > 0:
        for (per, tgt), g in df.groupby(["period", "target"], sort=False):
            top = g.sort_values("score_local", ascending=False).head(int(topk_per_pair))
            marks.extend([(f, per, tgt) for f in top["feature"].astype(str).tolist()])
    appear_rate = None
    if marks:
        mdf = pd.DataFrame(marks, columns=["feature", "period", "target"]).drop_duplicates()
        pair_count = max(1, int(mdf.groupby(["period", "target"]).ngroups))
        ar = mdf.groupby("feature").size().rename("appear_count").reset_index()
        ar["appear_rate"] = ar["appear_count"].astype(float) / float(pair_count)
        appear_rate = ar[["feature", "appear_rate"]]

    score_tp = df.groupby(["feature", "target"], sort=False)["score_local"].mean().rename("score_tp").reset_index()
    weights = _load_weights(weights_yaml)
    target_list = sorted(score_tp["target"].unique().tolist())
    if weights and len(weights) == len(target_list):
        w_map: Dict[str, float] = {t: float(w) for t, w in zip(target_list, weights)}
    else:
        w_map = {t: 1.0 for t in target_list}
    score_tp["w"] = score_tp["target"].map(w_map).astype(float)
    agg = score_tp.groupby("feature", sort=False).apply(
        lambda gdf: pd.Series(
            {
                "score_global": float((gdf["score_tp"] * gdf["w"]).sum() / max(1e-8, gdf["w"].sum())),
            }
        )
    ).reset_index()

    rank_stats = df.groupby("feature", sort=False)["rank_pct"].agg(
        avg_rank_pct="mean",
        std_rank_pct="std",
        pair_coverage="count",
    ).reset_index()
    agg = agg.merge(rank_stats, on="feature", how="left")

    if tft_file and os.path.exists(tft_file) and tft_bonus > 0:
        tft = pd.read_csv(tft_file)
        if {"feature", "score"}.issubset(tft.columns):
            tft = tft.set_index("feature")["score"]
            tft = (tft - tft.min()) / max(1e-8, (tft.max() - tft.min()))
            agg["score_global"] = agg.apply(
                lambda r: r["score_global"] + tft_bonus * float(tft.get(r["feature"], 0.0)),
                axis=1,
            )

    if appear_rate is not None and not appear_rate.empty:
        agg = agg.merge(appear_rate, on="feature", how="left")
        agg["appear_rate"] = agg["appear_rate"].fillna(0.0)
        if isinstance(min_appear_rate, (int, float)) and 0 <= float(min_appear_rate) <= 1:
            agg = agg[agg["appear_rate"] >= float(min_appear_rate)].copy()
    else:
        agg["appear_rate"] = np.nan

    agg = agg.sort_values("score_global", ascending=False).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)
    agg["keep"] = agg["rank"] <= int(topk)
    return agg


def export_selected_features(df: pd.DataFrame, out_txt: str) -> None:
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    keep = df[df["keep"]]["feature"].astype(str).tolist()
    with open(out_txt, "w", encoding="utf-8") as f:
        for c in keep:
            f.write(f"{c}\n")
    print(f"[save] {out_txt} ({len(keep)} features)")


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-period importances and produce unified core set")
    ap.add_argument("--in", type=str, default="reports/feature_evidence/tree_perm")
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--tft-file", type=str, default=None)
    ap.add_argument("--tft-bonus", type=float, default=0.0)
    ap.add_argument("--topk-per-pair", type=int, default=0)
    ap.add_argument("--min-appear-rate", type=float, default=0.0)
    ap.add_argument("--out-summary", type=str, default="reports/feature_evidence/aggregated_core.csv")
    ap.add_argument("--out-allowlist", type=str, default="configs/selected_features.txt")
    args = ap.parse_args()
    pair_df = aggregate_tree_perm(args.__dict__["in"])  # avoid keyword clash with Python reserved word
    core = build_unified_core(
        pair_df,
        args.weights,
        topk=args.topk,
        tft_bonus=args.tft_bonus,
        tft_file=args.tft_file,
        topk_per_pair=(args.topk_per_pair if args.topk_per_pair and args.topk_per_pair > 0 else None),
        min_appear_rate=(args.min_appear_rate if args.min_appear_rate and args.min_appear_rate > 0 else None),
    )
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    core.to_csv(args.out_summary, index=False)
    print(f"[save] {args.out_summary}")
    export_selected_features(core, args.out_allowlist)


if __name__ == "__main__":
    main()
