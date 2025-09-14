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
    ap.add_argument("--pkl", type=str, default=None, help="Override merged dataset PKL path")
    ap.add_argument("--time-perm", action="store_true")
    ap.add_argument("--block-len", type=int, default=36)
    ap.add_argument("--perm-repeats", type=int, default=5)
    ap.add_argument("--group-cols", type=str, default="symbol")
    ap.add_argument("--topk-per-pair", type=int, default=0)
    ap.add_argument("--min-appear-rate", type=float, default=0.0)
    ap.add_argument("--out-summary", type=str, default="reports/feature_evidence/aggregated_core.csv")
    ap.add_argument("--out-allowlist", type=str, default="configs/selected_features.txt")
    ap.add_argument("--allowlist", type=str, default=None, help="Optional feature allowlist (one name per line)")
    args = ap.parse_args()
    periods = args.periods.split(",") if args.periods else None
    if periods is None:
        ds = load_split(pkl_path=args.pkl or "data/pkl_merged/full_merged.pkl", val_mode=args.val_mode, val_days=args.val_days, val_ratio=args.val_ratio, allowlist_path=args.allowlist)
        periods = ds.periods
    gcols = [c.strip() for c in (args.group_cols or "").split(",") if c.strip()]
    run_tree_perm(periods, args.val_mode, args.val_days, args.val_ratio, args.preview, args.tree_out, allowlist_path=args.allowlist, pkl_path=args.pkl, time_perm=bool(args.time_perm), perm_method="cyclic_shift", block_len=int(args.block_len), group_cols=gcols, repeats=int(args.perm_repeats))
    agg = aggregate_tree_perm(args.tree_out)
    core = build_unified_core(agg, args.weights, topk=args.topk, tft_file=args.tft_gating, tft_bonus=args.tft_bonus, topk_per_pair=(args.topk_per_pair if args.topk_per_pair and args.topk_per_pair > 0 else None), min_appear_rate=(args.min_appear_rate if args.min_appear_rate and args.min_appear_rate > 0 else None))
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    core.to_csv(args.out_summary, index=False)
    export_selected_features(core, args.out_allowlist)
    print("[done] pipeline completed")


if __name__ == "__main__":
    main()
