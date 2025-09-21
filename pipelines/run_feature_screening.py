from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import yaml

import sys
CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from features.selection.run_pipeline import run_tree_perm, aggregate_tree_perm, build_unified_core, export_selected_features
from features.selection.common import load_split
from features.selection.filter_stage import run_filter_for_channel
from features.selection.embedded_stage import run_embedded_for_channel
from features.selection.optimize_subset import rfe_search, ga_search
from features.selection.combine_channels import combine_channels
from features.selection.rolling_validate import evaluate_post_validation

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _copy_if_needed(src: str, dst: str) -> None:
    if not os.path.exists(src):
        return
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    _ensure_dir(os.path.dirname(dst))
    shutil.copyfile(src, dst)

def _read_features(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def _as_list(val) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    return ["symbol"]

def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except Exception:
        return None

def _det_to_serializable(detail: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(detail, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in detail.items():
        if isinstance(v, dict):
            out[k] = {kk: _safe_float(vv) for kk, vv in v.items()}
        else:
            out[k] = _safe_float(v) if isinstance(v, (int, float)) or isinstance(v, str) else v
    return out

def main():
    ap = argparse.ArgumentParser(description="Run end-to-end feature screening (Base+Rich channels with gated fusion)")
    ap.add_argument("--config", type=str, default="pipelines/configs/feature_selection.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experts = cfg.get("experts", {})
    splits = cfg.get("splits", {})
    outer = splits.get("outer_val", {"mode": "days", "days": 90, "ratio": 0.2})
    perm = cfg.get("permutation", {}) or {}
    agg = cfg.get("aggregation", {}) or {}
    filter_cfg = cfg.get("filter", {}) or {}
    embedded_cfg = cfg.get("embedded", {}) or {}
    wrapper = cfg.get("wrapper", {}) or {}
    finalize = cfg.get("finalize", {}) or {}

    filter_enabled = bool(filter_cfg.get("enabled", True))
    filter_params_default = filter_cfg.get("params", {}) or {}
    filter_per_channel = filter_cfg.get("per_channel", {}) or {}

    embedded_enabled = bool(embedded_cfg.get("enabled", True))
    embedded_params_default = embedded_cfg.get("params", {}) or {}
    embedded_per_channel = embedded_cfg.get("per_channel", {}) or {}

    perm_enabled = bool(perm.get("enabled", True))
    block_len_by_period = perm.get("block_len_by_period", {}) or {}
    group_cols = _as_list(perm.get("group_cols", ["symbol"]))
    perm_repeats = int(perm.get("repeats", 5))
    default_block_len = int(perm.get("block_len", 36))
    perm_embargo = int(perm.get("embargo", 0))
    perm_purge = int(perm.get("purge", 0))
    perm_embargo_by = perm.get("embargo_by_period", {}) or {}
    perm_purge_by = perm.get("purge_by_period", {}) or {}
    perm_estimators = int(perm.get("n_estimators", 300))
    perm_seed = int(perm.get("seed", 42))

    weights_yaml = agg.get("weights_yaml")
    tree_out_root = "reports/feature_evidence"

    outputs_cfg = finalize.get("outputs", {}) or {}
    core_summary_csv = outputs_cfg.get("core_summary_csv", f"{tree_out_root}/aggregated_core.csv")
    core_allowlist_txt = outputs_cfg.get("core_allowlist_txt", "configs/selected_features.txt")
    optimized_allowlist_txt = outputs_cfg.get("optimized_allowlist_txt", f"{tree_out_root}/optimized_features.txt")
    plus_allowlist_txt = outputs_cfg.get("plus_allowlist_txt", f"{tree_out_root}/plus_features.txt")

    post_validation_cfg = finalize.get("post_validation", {}) or {}
    post_validation_enabled = bool(post_validation_cfg.get("enabled", False))

    documentation_cfg = finalize.get("documentation", {}) or {}
    doc_summary_enabled = documentation_cfg.get("summary", True) is not False

    expert_only_max = int(finalize.get("expert_only_max", 48))

    wrapper_ga_cfg = wrapper.get("ga", {}) or {}
    ga_multi_objective = bool(wrapper_ga_cfg.get("multi_objective", False))
    ga_weights_cfg = wrapper_ga_cfg.get("weights") or {}
    if isinstance(ga_weights_cfg, dict):
        ga_weights = {
            "classification": float(ga_weights_cfg.get("classification", 1.0)),
            "regression": float(ga_weights_cfg.get("regression", 1.0)),
        }
    else:
        ga_weights = {"classification": 1.0, "regression": 1.0}

    keys = list(experts.keys())
    total = len(keys)
    print(f"[pipeline] experts={total} outer=({outer.get('mode')}:{outer.get('days') or outer.get('ratio')}) time_perm={perm_enabled}")

    for idx, key in enumerate(keys, start=1):
        ex = experts[key]
        name = ex.get("name", key)
        pkl_base = ex.get("pkl_base", ex.get("pkl_path"))
        pkl_rich = ex.get("pkl_rich")
        periods = ex.get("periods", ["1h", "4h", "1d"]) if isinstance(ex.get("periods"), list) else ["1h", "4h", "1d"]

        filter_params_base = {**filter_params_default, **(filter_per_channel.get("base") or {})}
        filter_params_rich = {**filter_params_default, **(filter_per_channel.get("rich") or {})}
        embedded_params_base = {**embedded_params_default, **(embedded_per_channel.get("base") or {})}
        embedded_params_rich = {**embedded_params_default, **(embedded_per_channel.get("rich") or {})}

        block_len_map = block_len_by_period if isinstance(block_len_by_period, dict) else {}

        tree_out_base = os.path.join(tree_out_root, name, "base", "tree_perm")
        tree_out_rich = os.path.join(tree_out_root, name, "rich", "tree_perm")
        _ensure_dir(tree_out_base)
        _ensure_dir(tree_out_rich)

        print(f"\n[{idx}/{total}] Expert: {name} | periods={periods} | base={pkl_base} | rich={pkl_rich}")
        step_t0 = time.time()
        try:
            allow_base: Optional[str] = None
            allow_rich: Optional[str] = None
            res_base = None
            res_rich = None

            if filter_enabled and pkl_base:
                try:
                    res_base = run_filter_for_channel(
                        expert_name=name,
                        channel="base",
                        pkl_path=pkl_base,
                        val_mode=outer.get("mode", "days"),
                        val_days=int(outer.get("days", 90)),
                        val_ratio=float(outer.get("ratio", 0.2)),
                        allowlist_path=None,
                        params_dict=filter_params_base,
                    )
                    allow_base = str(res_base.allowlist_path)
                    print(f"  [filter] base keep={len(res_base.keep_features)}")
                except Exception as fe:  # pragma: no cover - defensive
                    print(f"  [filter-warn] base channel failed: {fe}")
            if filter_enabled and pkl_rich:
                try:
                    res_rich = run_filter_for_channel(
                        expert_name=name,
                        channel="rich",
                        pkl_path=pkl_rich,
                        val_mode=outer.get("mode", "days"),
                        val_days=int(outer.get("days", 90)),
                        val_ratio=float(outer.get("ratio", 0.2)),
                        allowlist_path=None,
                        params_dict=filter_params_rich,
                    )
                    allow_rich = str(res_rich.allowlist_path)
                    print(f"  [filter] rich keep={len(res_rich.keep_features)}")
                except Exception as fe:  # pragma: no cover - defensive
                    print(f"  [filter-warn] rich channel failed: {fe}")

            embedded_base_summary = None
            embedded_rich_summary = None
            if embedded_enabled and pkl_base:
                try:
                    emb_base = run_embedded_for_channel(
                        expert_name=name,
                        channel="base",
                        pkl_path=pkl_base,
                        val_mode=outer.get("mode", "days"),
                        val_days=int(outer.get("days", 90)),
                        val_ratio=float(outer.get("ratio", 0.2)),
                        allowlist_path=allow_base,
                        cfg=embedded_params_base,
                    )
                    embedded_base_summary = emb_base.summary
                    print(f"  [embedded] base summary_rows={len(emb_base.summary)}")
                except Exception as ee:  # pragma: no cover - defensive
                    print(f"  [embedded-warn] base channel failed: {ee}")
            if embedded_enabled and pkl_rich:
                try:
                    emb_rich = run_embedded_for_channel(
                        expert_name=name,
                        channel="rich",
                        pkl_path=pkl_rich,
                        val_mode=outer.get("mode", "days"),
                        val_days=int(outer.get("days", 90)),
                        val_ratio=float(outer.get("ratio", 0.2)),
                        allowlist_path=allow_rich,
                        cfg=embedded_params_rich,
                    )
                    embedded_rich_summary = emb_rich.summary
                    print(f"  [embedded] rich summary_rows={len(emb_rich.summary)}")
                except Exception as ee:  # pragma: no cover - defensive
                    print(f"  [embedded-warn] rich channel failed: {ee}")

            summary_base = None
            summary_rich = None

            # Step 1: base tree+perm
            print("  Step 1/4: base tree+perm ...", end="", flush=True)
            summary_base = run_tree_perm(
                periods=periods,
                val_mode=outer.get("mode", "days"),
                val_days=int(outer.get("days", 90)),
                val_ratio=float(outer.get("ratio", 0.2)),
                topn_preview=3,
                out_dir=tree_out_base,
                allowlist_path=allow_base,
                pkl_path=pkl_base,
                time_perm=bool(perm_enabled),
                perm_method=str(perm.get("method", "cyclic_shift")),
                block_len=default_block_len,
                block_len_by_period=block_len_map,
                group_cols=group_cols,
                repeats=perm_repeats,
                targets_override=[t for t in (ex.get("targets") or [])],
                embargo=perm_embargo,
                embargo_by_period=perm_embargo_by,
                purge=perm_purge,
                purge_by_period=perm_purge_by,
                n_estimators=perm_estimators,
                random_state=perm_seed,
            )
            print(" done")

            ran_rich = False
            if pkl_rich:
                print("  Step 2/4: rich tree+perm ...", end="", flush=True)
                summary_rich = run_tree_perm(
                    periods=periods,
                    val_mode=outer.get("mode", "days"),
                    val_days=int(outer.get("days", 90)),
                    val_ratio=float(outer.get("ratio", 0.2)),
                    topn_preview=3,
                    out_dir=tree_out_rich,
                    allowlist_path=allow_rich,
                    pkl_path=pkl_rich,
                    time_perm=bool(perm_enabled),
                    perm_method=str(perm.get("method", "cyclic_shift")),
                    block_len=default_block_len,
                    block_len_by_period=block_len_map,
                    group_cols=group_cols,
                    repeats=perm_repeats,
                    targets_override=[t for t in (ex.get("targets") or [])],
                    embargo=perm_embargo,
                    embargo_by_period=perm_embargo_by,
                    purge=perm_purge,
                    purge_by_period=perm_purge_by,
                    n_estimators=perm_estimators,
                    random_state=perm_seed + 17,
                )
                print(" done"); ran_rich = True

            # Step 3: combine core
            print("  Step 3/4: combine core ...", end="", flush=True)
            pair_df = None
            if ran_rich:
                core_df = combine_channels(
                    base_dir=tree_out_base,
                    rich_dir=tree_out_rich,
                    weights_yaml=weights_yaml,
                    topk=int(agg.get("topk_core", 128)),
                    topk_per_pair=int(agg.get("topk_per_pair", 64)) or None,
                    min_appear_rate=float(agg.get("min_appear_rate", 0.5)) or None,
                )
            else:
                pair_df = aggregate_tree_perm(tree_out_base)
                core_df = build_unified_core(
                    pair_df,
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
            core_df.to_csv(core_out_csv, index=False)
            export_path = os.path.join(core_dir, Path(core_allowlist_txt).name)
            export_selected_features(core_df, export_path)
            core_std = os.path.join(core_dir, "core_allowlist.txt")
            _copy_if_needed(export_path, core_std)
            kept = int(core_df[core_df["keep"]].shape[0])
            print(f" done (kept={kept})")

            # Step 4: wrapper search
            print("  Step 4/4: wrapper search ...", end="", flush=True)
            ds = load_split(
                pkl_path=pkl_rich or pkl_base,
                val_mode=outer.get("mode", "days"),
                val_days=int(outer.get("days", 90)),
                val_ratio=float(outer.get("ratio", 0.2)),
                allowlist_path=export_path,
            )
            core_feats = core_df[core_df["keep"]]["feature"].astype(str).tolist()
            pool = core_feats.copy()
            method = (wrapper.get("method", "rfe") or "rfe").lower()
            if method == "ga":
                feats, score, det = ga_search(
                    pool,
                    ds.train,
                    ds.val,
                    ds.targets,
                    weights_yaml,
                    pop_size=int(wrapper_ga_cfg.get("pop", 30)),
                    generations=int(wrapper_ga_cfg.get("gen", 15)),
                    cx_prob=float(wrapper_ga_cfg.get("cx", 0.7)),
                    mut_prob=float(wrapper_ga_cfg.get("mut", 0.1)),
                    sample_cap=int(wrapper_ga_cfg.get("sample_cap", 50000)),
                    multi_objective=ga_multi_objective,
                    mo_weights=ga_weights,
                )
            else:
                feats, score, det = rfe_search(
                    pool,
                    ds.train,
                    ds.val,
                    ds.targets,
                    weights_yaml,
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
            plus_feats = extras[:expert_only_max] if expert_only_max > 0 else extras
            plus_out = os.path.join(opt_dir, Path(plus_allowlist_txt).name)
            with open(plus_out, "w", encoding="utf-8") as f:
                for c in plus_feats:
                    f.write(str(c).strip() + "\n")
            combined_feats = core_feats + plus_feats
            final_out = os.path.join(opt_dir, f"allowlist_{name}.txt")
            with open(final_out, "w", encoding="utf-8") as f:
                for c in combined_feats:
                    f.write(str(c).strip() + "\n")
            print(f" done (score={score:.6f}, core={len(core_feats)}, plus={len(plus_feats)}, final={len(combined_feats)})")

            post_summary = None
            if post_validation_enabled:
                pv_dir = os.path.join(opt_dir, "post_validation")
                post_summary = evaluate_post_validation(
                    combined_feats,
                    ds,
                    post_validation_cfg,
                    weights_yaml,
                    periods=periods,
                    out_dir=pv_dir,
                )
                mean_score = _safe_float(post_summary.get("mean_score"))
                threshold = _safe_float(post_validation_cfg.get("min_score"))
                if threshold is not None and mean_score is not None and mean_score < threshold:
                    print(f"  [post-warn] mean_score={mean_score:.6f} < threshold={threshold:.6f}")

            if doc_summary_enabled:
                summary_dict: Dict[str, Any] = {
                    "expert": name,
                    "periods": periods,
                    "core": {
                        "core_count": len(core_feats),
                        "plus_count": len(plus_feats),
                        "final_count": len(combined_feats),
                        "wrapper_method": method,
                        "wrapper_score": _safe_float(score),
                        "wrapper_detail": _det_to_serializable(det),
                        "core_summary_path": core_out_csv,
                        "core_allowlist_path": export_path,
                        "plus_allowlist_path": plus_out,
                        "optimized_allowlist_path": opt_out_txt,
                    },
                }
                if post_summary:
                    summary_dict["post_validation"] = post_summary
                channel_info: Dict[str, Any] = {}
                if res_base:
                    base_dir = Path(tree_out_root) / name / "base"
                    channel_info["base"] = {
                        "filter_kept": len(res_base.keep_features),
                        "filter_dropped": int(res_base.dropped.shape[0]) if isinstance(res_base.dropped, pd.DataFrame) else None,
                        "filter_allowlist": str(res_base.allowlist_path),
                        "filter_stats": str(Path(res_base.allowlist_path).parent / "filter_stats.csv"),
                        "embedded_rows": int(embedded_base_summary.shape[0]) if embedded_base_summary is not None else None,
                        "embedded_summary": str(base_dir / "stage2_embedded" / "summary.csv"),
                        "tree_perm_summary": str(base_dir / "tree_perm" / "summary.csv"),
                    }
                if res_rich:
                    rich_dir = Path(tree_out_root) / name / "rich"
                    channel_info["rich"] = {
                        "filter_kept": len(res_rich.keep_features),
                        "filter_dropped": int(res_rich.dropped.shape[0]) if isinstance(res_rich.dropped, pd.DataFrame) else None,
                        "filter_allowlist": str(res_rich.allowlist_path),
                        "filter_stats": str(Path(res_rich.allowlist_path).parent / "filter_stats.csv"),
                        "embedded_rows": int(embedded_rich_summary.shape[0]) if embedded_rich_summary is not None else None,
                        "embedded_summary": str(rich_dir / "stage2_embedded" / "summary.csv"),
                        "tree_perm_summary": str(rich_dir / "tree_perm" / "summary.csv"),
                    }
                if channel_info:
                    summary_dict["channels"] = channel_info
                summary_path = Path(tree_out_root) / name / "summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with summary_path.open("w", encoding="utf-8") as f:
                    json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"\n  [error] expert={name} failed: {e}")
        finally:
            dt = time.time() - step_t0
            print(f"  [time] expert={name} elapsed={dt:.1f}s")

    try:
        core_lists = []
        for key, ex in experts.items():
            name = ex.get("name", key)
            core_dir = os.path.join(tree_out_root, name)
            export_path = os.path.join(core_dir, Path(core_allowlist_txt).name)
            feats = _read_features(export_path)
            if feats:
                core_lists.append(set(feats))
        if core_lists:
            core_common = set.intersection(*core_lists) if len(core_lists) > 1 else core_lists[0]
            common_out = os.path.join(tree_out_root, "allowlist_core_common.txt")
            with open(common_out, "w", encoding="utf-8") as f:
                for c in sorted(core_common):
                    f.write(f"{c}\n")
            print(f"[save] unified core (intersection) -> {common_out} ({len(core_common)})")
    except Exception as e:
        print(f"[warn] failed to compose unified core: {e}")

    print("\n[done] feature screening for all configured experts")


if __name__ == "__main__":
    main()
