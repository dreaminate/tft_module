from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from typing import List

import yaml

import sys
CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from features.selection.run_pipeline import run_tree_perm, aggregate_tree_perm, build_unified_core, export_selected_features
from features.selection.common import load_split
from features.selection.optimize_subset import rfe_search, ga_search
from features.selection.combine_channels import combine_channels


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_features(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def main():
    ap = argparse.ArgumentParser(description="Run end-to-end feature screening (Base+Rich channels with gated fusion)")
    ap.add_argument("--config", type=str, default="pipelines/configs/feature_selection.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experts = cfg.get("experts", {})
    splits = cfg.get("splits", {})
    outer = splits.get("outer_val", {"mode": "days", "days": 90, "ratio": 0.2})
    perm = cfg.get("permutation", {})
    agg = cfg.get("aggregation", {})
    wrapper = cfg.get("wrapper", {})
    finalize = cfg.get("finalize", {})

    weights_yaml = agg.get("weights_yaml", "configs/weights_config.yaml")
    tree_out_root = "reports/feature_evidence"
    core_summary_csv = finalize.get("outputs", {}).get("core_summary_csv", f"{tree_out_root}/aggregated_core.csv")
    core_allowlist_txt = finalize.get("outputs", {}).get("core_allowlist_txt", "configs/selected_features.txt")
    optimized_allowlist_txt = finalize.get("outputs", {}).get("optimized_allowlist_txt", f"{tree_out_root}/optimized_features.txt")

    keys = list(experts.keys())
    total = len(keys)
    print(f"[pipeline] experts={total} outer=({outer.get('mode')}:{outer.get('days') or outer.get('ratio')}) time_perm={perm.get('enabled', True)}")

    for idx, key in enumerate(keys, start=1):
        ex = experts[key]
        name = ex.get("name", key)
        pkl_base = ex.get("pkl_base", ex.get("pkl_path"))
        pkl_rich = ex.get("pkl_rich")
        periods = ex.get("periods", ["1h", "4h", "1d"]) if isinstance(ex.get("periods"), list) else ["1h", "4h", "1d"]

        block_len_by_period = perm.get("block_len_by_period", {})
        tree_out_base = os.path.join(tree_out_root, name, "base", "tree_perm")
        tree_out_rich = os.path.join(tree_out_root, name, "rich", "tree_perm")
        _ensure_dir(tree_out_base); _ensure_dir(tree_out_rich)

        print(f"\n[{idx}/{total}] Expert: {name} | periods={periods} | base={pkl_base} | rich={pkl_rich}")
        step_t0 = time.time()
        try:
            # Step 1: base channel
            print("  Step 1/4: base tree+perm ...", end="", flush=True)
            run_tree_perm(
                periods=periods,
                val_mode=outer.get("mode", "days"),
                val_days=int(outer.get("days", 90)),
                val_ratio=float(outer.get("ratio", 0.2)),
                topn_preview=3,
                out_dir=tree_out_base,
                allowlist_path=None,
                pkl_path=pkl_base,
                time_perm=bool(perm.get("enabled", True)),
                perm_method=str(perm.get("method", "cyclic_shift")),
                block_len=int(block_len_by_period.get(periods[0], 36)),
                block_len_by_period=block_len_by_period,
                group_cols=list(perm.get("group_cols", ["symbol"])),
                repeats=int(perm.get("repeats", 5)),
                targets_override=[t for t in ex.get("targets", [])],
            )
            print(" done")

            # Step 2: rich channel
            ran_rich = False
            if pkl_rich:
                print("  Step 2/4: rich tree+perm ...", end="", flush=True)
                run_tree_perm(
                    periods=periods,
                    val_mode=outer.get("mode", "days"),
                    val_days=int(outer.get("days", 90)),
                    val_ratio=float(outer.get("ratio", 0.2)),
                    topn_preview=3,
                    out_dir=tree_out_rich,
                    allowlist_path=None,
                    pkl_path=pkl_rich,
                    time_perm=bool(perm.get("enabled", True)),
                    perm_method=str(perm.get("method", "cyclic_shift")),
                    block_len=int(block_len_by_period.get(periods[0], 36)),
                    block_len_by_period=block_len_by_period,
                    group_cols=list(perm.get("group_cols", ["symbol"])),
                    repeats=int(perm.get("repeats", 5)),
                    targets_override=[t for t in ex.get("targets", [])],
                )
                print(" done"); ran_rich = True

            # Step 3: combine/aggregate core
            print("  Step 3/4: combine core ...", end="", flush=True)
            if ran_rich:
                core = combine_channels(
                    base_dir=tree_out_base,
                    rich_dir=tree_out_rich,
                    weights_yaml=weights_yaml,
                    topk=int(agg.get("topk_core", 128)),
                    topk_per_pair=int(agg.get("topk_per_pair", 64)) or None,
                    min_appear_rate=float(agg.get("min_appear_rate", 0.5)) or None,
                )
            else:
                agg_df = aggregate_tree_perm(tree_out_base)
                core = build_unified_core(
                    agg_df,
                    weights_yaml,
                    topk=int(agg.get("topk_core", 128)),
                    tft_bonus=0.0,
                    tft_file=None,
                    topk_per_pair=int(agg.get("topk_per_pair", 0)) or None,
                    min_appear_rate=float(agg.get("min_appear_rate", 0.0)) or None,
                )
            core_dir = os.path.join(tree_out_root, name)
            _ensure_dir(core_dir)
            core_out_csv = os.path.join(core_dir, Path(core_summary_csv).name)
            core_out_txt = os.path.join(core_dir, Path(core_allowlist_txt).name)
            core.to_csv(core_out_csv, index=False)
            export_selected_features(core, core_out_txt)
            kept = int(core[core["keep"]].shape[0])
            print(f" done (kept={kept})")

            # Step 4: wrapper search (use richer dataset if available)
            print("  Step 4/4: wrapper search ...", end="", flush=True)
            ds = load_split(pkl_path=pkl_rich or pkl_base, val_mode=outer.get("mode", "days"), val_days=int(outer.get("days", 90)), val_ratio=float(outer.get("ratio", 0.2)))
            core_feats = core[core["keep"]]["feature"].astype(str).tolist()
            pool = core_feats.copy()
            method = wrapper.get("method", "rfe").lower()
            if method == "ga":
                feats, score, det = ga_search(
                    pool, ds.train, ds.val, ds.targets, weights_yaml,
                    pop_size=int(wrapper.get("ga", {}).get("pop", 30)),
                    generations=int(wrapper.get("ga", {}).get("gen", 15)),
                    cx_prob=float(wrapper.get("ga", {}).get("cx", 0.7)),
                    mut_prob=float(wrapper.get("ga", {}).get("mut", 0.1)),
                    sample_cap=int(wrapper.get("ga", {}).get("sample_cap", 50000)),
                )
            else:
                feats, score, det = rfe_search(
                    pool, ds.train, ds.val, ds.targets, weights_yaml,
                    sample_cap=int(wrapper.get("rfe", {}).get("sample_cap", 50000)),
                    patience=int(wrapper.get("rfe", {}).get("patience", 10)),
                )
            opt_dir = core_dir
            _ensure_dir(opt_dir)
            opt_out_txt = os.path.join(opt_dir, Path(optimized_allowlist_txt).name)
            with open(opt_out_txt, "w", encoding="utf-8") as f:
                for c in feats:
                    f.write(str(c).strip() + "\n")
            core_set = set(core_feats)
            extras = [c for c in feats if c not in core_set]
            cap = int(finalize.get("expert_only_max", 48))
            final_feats = core_feats + extras[:cap]
            final_out = os.path.join(opt_dir, f"allowlist_{name}.txt")
            with open(final_out, "w", encoding="utf-8") as f:
                for c in final_feats:
                    f.write(str(c).strip() + "\n")
            print(f" done (score={score:.6f}, final={len(final_feats)})")
        except Exception as e:
            print(f"\n  [error] expert={name} failed: {e}")
        finally:
            dt = time.time() - step_t0
            print(f"  [time] expert={name} elapsed={dt:.1f}s")

    # Compose unified core across experts (intersection of per-expert core allowlists)
    try:
        core_lists = []
        for key, ex in experts.items():
            name = ex.get("name", key)
            core_dir = os.path.join(tree_out_root, name)
            core_out_txt = os.path.join(core_dir, Path(core_allowlist_txt).name)
            feats = _read_features(core_out_txt)
            if feats:
                core_lists.append(set(feats))
        if core_lists:
            core_common = set.intersection(*core_lists) if len(core_lists) > 1 else core_lists[0]
            common_out = os.path.join(tree_out_root, "allowlist_core_common.txt")
            with open(common_out, "w", encoding="utf-8") as f:
                for c in sorted(core_common):
                    f.write(str(c).strip() + "\n")
            print(f"[save] unified core (intersection) -> {common_out} ({len(core_common)})")
    except Exception as e:
        print(f"[warn] failed to compose unified core: {e}")

    print("\n[done] feature screening for all configured experts")


if __name__ == "__main__":
    main()
