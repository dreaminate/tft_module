from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import yaml


def _rank01(series: pd.Series, ascending: bool = False) -> pd.Series:
    return series.rank(method="average", ascending=ascending, na_option="keep").astype(float) / series.count()


def load_weights(yaml_path: str) -> List[float]:
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ws = cfg.get("custom_weights", [])
    return [float(w) for w in ws]


def aggregate_tree_perm(in_dir: str) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for per in sorted(os.listdir(in_dir)):
        per_dir = Path(in_dir) / per
        if not per_dir.is_dir():
            continue
        for csv in per_dir.glob("*_importances.csv"):
            target = csv.stem.replace("_importances", "")
            df = pd.read_csv(csv)
            s_tree = _rank01(df["tree_importance"], ascending=False)
            s_perm = _rank01(df["perm_mean"], ascending=False)
            sub = pd.DataFrame({
                "feature": df["feature"],
                "period": str(per),
                "target": target,
                "score_tree": s_tree,
                "score_perm": s_perm,
            })
            rows.append(sub)
    if not rows:
        raise SystemExit(f"[err] no importances found under {in_dir}")
    return pd.concat(rows, ignore_index=True)


def build_unified_core(
    df: pd.DataFrame,
    weights_yaml: str,
    topk: int = 128,
    tft_bonus: float = 0.0,
    tft_file: str | None = None,
    topk_per_pair: int | None = None,
    min_appear_rate: float | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df["score_local"] = 0.5 * df["score_tree"].fillna(0) + 0.5 * df["score_perm"].fillna(0)
    # appearance rate across (period,target) pairs
    appear_rate = None
    if topk_per_pair is not None and topk_per_pair > 0:
        marks = []
        for (per, tgt), g in df.groupby(["period", "target"], sort=False):
            g = g.sort_values("score_local", ascending=False)
            topk_sub = g.head(int(topk_per_pair))["feature"].astype(str).tolist()
            marks.extend([(f, per, tgt) for f in topk_sub])
        if marks:
            mdf = pd.DataFrame(marks, columns=["feature", "period", "target"]).drop_duplicates()
            num_pairs = max(1, int(mdf.groupby(["period", "target"]).ngroups))
            appear = mdf.groupby("feature").size().rename("appear_count").reset_index()
            appear["appear_rate"] = appear["appear_count"].astype(float) / float(num_pairs)
            appear_rate = appear[["feature", "appear_rate"]]
        else:
            appear_rate = pd.DataFrame({"feature": [], "appear_rate": []})

    ft = df.groupby(["feature", "target"], as_index=False)["score_local"].mean().rename(columns={"score_local": "score_tp"})
    weights = load_weights(weights_yaml)
    target_list = sorted(ft["target"].unique().tolist())
    if weights and len(weights) == len(target_list):
        w_map = {t: float(w) for t, w in zip(target_list, weights)}
    else:
        w_map = {t: 1.0 for t in target_list}
    ft["w"] = ft["target"].map(w_map).astype(float)
    g = ft.groupby("feature", as_index=False).apply(lambda gdf: pd.Series({
        "score_global": float((gdf["score_tp"] * gdf["w"]).sum() / max(1e-8, gdf["w"].sum()))
    })).reset_index()
    if tft_file and os.path.exists(tft_file) and tft_bonus > 0:
        tft = pd.read_csv(tft_file)
        if "feature" in tft.columns and "score" in tft.columns:
            tft_score = (tft.set_index("feature")["score"] - tft["score"].min()) / max(1e-8, (tft["score"].max() - tft["score"].min()))
            g["score_global"] = g.apply(lambda r: r["score_global"] + tft_bonus * float(tft_score.get(r["feature"], 0.0)), axis=1)
    # merge appearance rate and filter if requested
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


def export_selected_features(df: pd.DataFrame, out_txt: str) -> None:
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    keep = df[df["keep"]]["feature"].tolist()
    with open(out_txt, "w", encoding="utf-8") as f:
        for c in keep:
            f.write(str(c).strip() + "\n")
    print(f"[save] {out_txt} ({len(keep)} features)")


def main():
    ap = argparse.ArgumentParser(description="Aggregate per-period importances and produce unified core set")
    ap.add_argument("--in", type=str, default="reports/feature_evidence/tree_perm")
    ap.add_argument("--weights", type=str, default="configs/weights_config.yaml")
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--tft-file", type=str, default=None)
    ap.add_argument("--tft-bonus", type=float, default=0.0)
    ap.add_argument("--topk-per-pair", type=int, default=0)
    ap.add_argument("--min-appear-rate", type=float, default=0.0)
    ap.add_argument("--out-summary", type=str, default="reports/feature_evidence/aggregated_core.csv")
    ap.add_argument("--out-allowlist", type=str, default="configs/selected_features.txt")
    args = ap.parse_args()
    df = aggregate_tree_perm(args.__dict__["in"])  # avoid keyword
    core = build_unified_core(
        df,
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
