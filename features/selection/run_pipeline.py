from __future__ import annotations
import argparse
import os
from .tree_perm import run as run_tree_perm
from .aggregate_core import aggregate_tree_perm, build_unified_core, export_selected_features
from .common import load_split


def main():
    ap = argparse.ArgumentParser(description="Feature screening pipeline: tree+perm -> aggregate -> core set")
    ap.add_argument("--periods", type=str, default=None)
    ap.add_argument("--val-mode", type=str, default="ratio", choices=["ratio","days"]) 
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--preview", type=int, default=10)
    ap.add_argument("--tree-out", type=str, default="reports/feature_evidence/tree_perm")
    ap.add_argument("--weights", type=str, default="configs/weights_config.yaml")
    ap.add_argument("--tft-gating", type=str, default=None)
    ap.add_argument("--tft-bonus", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--out-summary", type=str, default="reports/feature_evidence/aggregated_core.csv")
    ap.add_argument("--out-allowlist", type=str, default="configs/selected_features.txt")
    args = ap.parse_args()
    periods = args.periods.split(",") if args.periods else None
    if periods is None:
        ds = load_split(val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio)
        periods = ds.periods
    run_tree_perm(periods, args.val_mode, args.val_days, args.val_ratio, args.preview, args.tree_out)
    agg = aggregate_tree_perm(args.tree_out)
    core = build_unified_core(agg, args.weights, topk=args.topk, tft_file=args.tft_gating, tft_bonus=args.tft_bonus)
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    core.to_csv(args.out_summary, index=False)
    export_selected_features(core, args.out_allowlist)
    print("[done] pipeline completed")


if __name__ == "__main__":
    main()

