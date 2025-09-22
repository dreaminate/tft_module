from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import yaml

import sys
CURR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURR_DIR, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from features.selection.run_pipeline import run_tree_perm, aggregate_tree_perm, build_unified_core, export_selected_features
from features.selection.common import load_split, TARGETS as _ALL_TARGETS, REGRESSION_TARGETS as _REG_TARGETS
from features.selection.optimize_subset import ga_search_multi
from features.selection.rolling_validate import _evaluate as _eval_once
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


def _robust_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out.get("symbol").astype(str)
    out["period"] = out.get("period").astype(str)
    if "datetime" in out.columns:
        try:
            out["datetime"] = pd.to_datetime(out["datetime"], format="mixed", errors="coerce")
        except TypeError:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    if out.get("datetime") is None or out["datetime"].isna().all():
        if "timestamp" in out.columns:
            ts = pd.to_numeric(out["timestamp"], errors="coerce")
            med = float(ts.dropna().median()) if ts.dropna().size else 0.0
            unit = "ms" if med >= 1e11 else "s"
            out["datetime"] = pd.to_datetime(ts, unit=unit, errors="coerce")
        else:
            out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out


def _build_intersect_dataset(
    expert_name: str,
    pkl_base: str,
    pkl_rich: str,
    out_root: str,
) -> Optional[str]:
    try:
        if not (pkl_base and os.path.exists(pkl_base) and pkl_rich and os.path.exists(pkl_rich)):
            return None
        dfb = pd.read_pickle(pkl_base)
        dfr = pd.read_pickle(pkl_rich)
        dfb = _robust_datetime(dfb)
        dfr = _robust_datetime(dfr)
        # essential columns
        key_cols = [c for c in ["symbol", "period", "datetime"] if c in dfb.columns or c in dfr.columns]
        tgt_cols = [t for t in _ALL_TARGETS if t in dfb.columns or t in dfr.columns]
        reserved = set(key_cols + ["timestamp", "time_idx"]) | set(tgt_cols)
        # intersect feature columns
        feat_b = set([c for c in dfb.columns if c not in reserved])
        feat_r = set([c for c in dfr.columns if c not in reserved])
        feats = sorted(list(feat_b & feat_r))
        if not feats:
            return None
        # intersect rows by keys
        k_b = dfb[key_cols].apply(tuple, axis=1)
        k_r = dfr[key_cols].apply(tuple, axis=1)
        inter = set(k_b) & set(k_r)
        if not inter:
            return None
        mask_b = k_b.isin(inter)
        dfb_i = dfb.loc[mask_b, key_cols + feats].copy()
        # bring targets from base, fill missing from rich
        for t in tgt_cols:
            if t in dfb.columns:
                dfb_i[t] = dfb.loc[mask_b, t].values
        # fill missing targets from rich by merging
        missing_t = [t for t in tgt_cols if t not in dfb_i.columns]
        if missing_t:
            dfr_keys = dfr[key_cols + missing_t].copy()
            dfb_i = dfb_i.merge(dfr_keys, on=key_cols, how="left")
        # sort and write
        dfb_i = dfb_i.sort_values(key_cols)
        out_dir = os.path.join(out_root, f"{expert_name}_intersect")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "full_merged_with_fundamentals.pkl")
        dfb_i.to_pickle(out_path)
        return out_path
    except Exception:
        return None


def _ensure_dirpath(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _ensure_global_onchain_datasets() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"base_pkl": None, "rich_pkl": None, "onchain_only_pkl": None, "allowlist_base": None, "allowlist_rich": None}
    # locate full_merged (CSV/PKL)
    base_csv = os.path.join("data", "merged", "full_merged.csv")
    base_pkl = os.path.join("data", "merged", "full_merged.pkl")
    df = None
    try:
        if os.path.exists(base_pkl):
            df = pd.read_pickle(base_pkl)
        elif os.path.exists(base_csv):
            df = pd.read_csv(base_csv)
            _ensure_dirpath(base_pkl)
            df.to_pickle(base_pkl)
        else:
            return out
        df = _robust_datetime(df)
        # identify columns
        key_cols = [c for c in ["symbol", "period", "datetime"] if c in df.columns]
        tgt_cols = [t for t in _ALL_TARGETS if t in df.columns]
        # classify columns by prefix
        def is_onchain(name: str) -> bool:
            return str(name).startswith("onchain_") or str(name).startswith("etf_")
        feat_cols = [c for c in df.columns if c not in set(key_cols + ["timestamp", "time_idx"]) and not str(c).startswith("target_")]
        base_feats = [c for c in feat_cols if not is_onchain(c)]
        rich_feats = [c for c in feat_cols if is_onchain(c)]
        # write base allowlist
        allow_base = os.path.join("data", "merged", "allowlist_base.txt")
        _ensure_dirpath(allow_base)
        with open(allow_base, "w", encoding="utf-8") as f:
            for c in base_feats:
                f.write(f"{c}\n")
        out["allowlist_base"] = allow_base
        # build full_onchain_merged as max-row-intersection on rich features (drop rows with any NaN in rich features)
        if rich_feats:
            df_on = df.copy()
            # rows where all rich features are available
            mask = (~df_on[rich_feats].isna()).all(axis=1)
            df_on = df_on.loc[mask, key_cols + tgt_cols + rich_feats]
            on_csv = os.path.join("data", "merged", "full_onchain_merged.csv")
            on_pkl = os.path.join("data", "merged", "full_onchain_merged.pkl")
            _ensure_dirpath(on_csv)
            df_on.to_csv(on_csv, index=False)
            df_on.to_pickle(on_pkl)
            out["rich_pkl"] = on_pkl
            # write rich allowlist
            allow_rich = os.path.join("data", "merged", "allowlist_rich.txt")
            with open(allow_rich, "w", encoding="utf-8") as f:
                for c in rich_feats:
                    f.write(f"{c}\n")
            out["allowlist_rich"] = allow_rich
            # only_onchain_merged (only onchain + keys + targets)
            only_on = df.loc[:, key_cols + tgt_cols + rich_feats].copy()
            only_on_csv = os.path.join("data", "merged", "only_onchain_merged.csv")
            only_on_pkl = os.path.join("data", "merged", "only_onchain_merged.pkl")
            _ensure_dirpath(only_on_csv)
            only_on.to_csv(only_on_csv, index=False)
            only_on.to_pickle(only_on_pkl)
            out["onchain_only_pkl"] = only_on_pkl
        out["base_pkl"] = base_pkl
    except Exception:
        pass
    return out

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
    ap.add_argument("--experts", type=str, default=None, help="Comma-separated expert keys to run (override). Default: run all in config")
    ap.add_argument("--enable-base", action="store_true", help="Enable Base channel for selection")
    ap.add_argument("--enable-rich", action="store_true", help="Enable Rich channel for selection")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    experts_cfg = cfg.get("experts", {})
    # 允许通过 YAML 中的 experts_run 指定要运行的专家子集，CLI --experts 优先
    experts_run_yaml = cfg.get("experts_run")
    selected_keys = None
    if isinstance(args.experts, str) and args.experts:
        selected_keys = [k.strip() for k in args.experts.split(",") if k.strip()]
    elif isinstance(experts_run_yaml, list) and experts_run_yaml:
        selected_keys = [str(k) for k in experts_run_yaml]
    # 过滤出将要运行的专家字典
    if selected_keys:
        experts = {k: v for k, v in experts_cfg.items() if k in set(selected_keys)}
    else:
        experts = experts_cfg
    # 跳过标记为 skip_selection 的整合型专家（如 Regime 等）
    skipped = [k for k, ex in experts.items() if bool(ex.get("skip_selection", False))]
    if skipped:
        experts = {k: v for k, v in experts.items() if k not in set(skipped)}
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
    min_appear_rate_era = agg.get("min_appear_rate_era", None)
    rich_quality_weights = agg.get("rich_quality_weights", {}) or {}
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

    # ===== 计算专家 targets 并集 =====
    def _as_target_list(x):
        if isinstance(x, list):
            return [str(t) for t in x if t]
        return []
    union_targets = []
    allowed = set(_ALL_TARGETS)
    for _, ex in (experts or {}).items():
        for t in _as_target_list(ex.get("targets")):
            if t in allowed and t not in union_targets:
                union_targets.append(t)
    # 若未配置任何 expert 或其 targets 为空，则回退为全量内置 TARGETS
    if not union_targets:
        union_targets = list(_ALL_TARGETS)
    # 为 filter 阶段准备 ic/mi 目标（回归 -> IC，分类 -> MI）
    reg_set = set(_REG_TARGETS)
    ic_targets_union = [t for t in union_targets if t in reg_set]
    mi_targets_union = [t for t in union_targets if t not in reg_set]

    keys = list(experts.keys())
    total = len(keys)
    print(f"[pipeline] experts={total} outer=({outer.get('mode')}:{outer.get('days') or outer.get('ratio')}) time_perm={perm_enabled}")
    if skipped:
        print(f"[pipeline] skipped_experts={skipped} (skip_selection=true)")
    if selected_keys:
        print(f"[pipeline] selected_experts={selected_keys}")
    # Channel switches
    ch_cfg = cfg.get("channels", {}) or {}
    # default both true; CLI overrides per-flag when provided
    enable_base = bool(ch_cfg.get("base", True))
    enable_rich = bool(ch_cfg.get("rich", True))
    if args.enable_base:
        enable_base = True
        enable_rich = False if not args.enable_rich else enable_rich
    if args.enable_rich:
        enable_rich = True
        enable_base = False if not args.enable_base else enable_base
    print(f"[pipeline] channels: base={enable_base} rich={enable_rich}")

    # Ensure global onchain datasets/allowlists ready (for channel-specific selection)
    onchain_paths = _ensure_global_onchain_datasets()

    for idx, key in enumerate(keys, start=1):
        ex = experts[key]
        name = ex.get("name", key)
        pkl_base = ex.get("pkl_base", ex.get("pkl_path"))
        pkl_rich = ex.get("pkl_rich")
        # Build intersect dataset for selection when both channels exist
        intersect_path = None
        if pkl_base and pkl_rich and os.path.exists(pkl_base) and os.path.exists(pkl_rich):
            intersect_path = _build_intersect_dataset(name, pkl_base, pkl_rich, out_root=os.path.join("data", "merged", "expert_group"))
        periods = ex.get("periods", ["1h", "4h", "1d"]) if isinstance(ex.get("periods"), list) else ["1h", "4h", "1d"]

        filter_params_base = {**filter_params_default, **(filter_per_channel.get("base") or {})}
        filter_params_rich = {**filter_params_default, **(filter_per_channel.get("rich") or {})}
        # 若未指定 ic/mi 目标，使用并集拆分
        if not filter_params_base.get("ic_targets") and not filter_params_base.get("mi_targets"):
            filter_params_base = {
                **filter_params_base,
                "ic_targets": ic_targets_union,
                "mi_targets": mi_targets_union,
            }
        if not filter_params_rich.get("ic_targets") and not filter_params_rich.get("mi_targets"):
            filter_params_rich = {
                **filter_params_rich,
                "ic_targets": ic_targets_union,
                "mi_targets": mi_targets_union,
            }
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

            if enable_base and filter_enabled and pkl_base:
                try:
                    res_base = run_filter_for_channel(
                        expert_name=name,
                        channel="base",
                        pkl_path=onchain_paths.get("base_pkl") or pkl_base,
                        val_mode=outer.get("mode", "days"),
                        val_days=int(outer.get("days", 90)),
                        val_ratio=float(outer.get("ratio", 0.2)),
                        allowlist_path=onchain_paths.get("allowlist_base"),
                        params_dict=filter_params_base,
                    )
                    allow_base = str(res_base.allowlist_path)
                    print(f"  [filter] base keep={len(res_base.keep_features)}")
                except Exception as fe:  # pragma: no cover - defensive
                    print(f"  [filter-warn] base channel failed: {fe}")
            if enable_rich and filter_enabled and (pkl_rich or onchain_paths.get("rich_pkl")):
                try:
                    res_rich = run_filter_for_channel(
                        expert_name=name,
                        channel="rich",
                        pkl_path=onchain_paths.get("rich_pkl") or pkl_rich,
                        val_mode=outer.get("mode", "days"),
                        val_days=int(outer.get("days", 90)),
                        val_ratio=float(outer.get("ratio", 0.2)),
                        allowlist_path=onchain_paths.get("allowlist_rich"),
                        params_dict=filter_params_rich,
                    )
                    allow_rich = str(res_rich.allowlist_path)
                    print(f"  [filter] rich keep={len(res_rich.keep_features)}")
                except Exception as fe:  # pragma: no cover - defensive
                    print(f"  [filter-warn] rich channel failed: {fe}")

            embedded_base_summary = None
            embedded_rich_summary = None
            if enable_base and embedded_enabled and pkl_base:
                try:
                    emb_base = run_embedded_for_channel(
                        expert_name=name,
                        channel="base",
                        pkl_path=onchain_paths.get("base_pkl") or pkl_base,
                        val_mode=outer.get("mode", "days"),
                        val_days=int(outer.get("days", 90)),
                        val_ratio=float(outer.get("ratio", 0.2)),
                        allowlist_path=allow_base,
                        targets_override=union_targets,
                        cfg=embedded_params_base,
                    )
                    embedded_base_summary = emb_base.summary
                    print(f"  [embedded] base summary_rows={len(emb_base.summary)}")
                except Exception as ee:  # pragma: no cover - defensive
                    print(f"  [embedded-warn] base channel failed: {ee}")
            if enable_rich and embedded_enabled and (pkl_rich or onchain_paths.get("rich_pkl")):
                try:
                    emb_rich = run_embedded_for_channel(
                        expert_name=name,
                        channel="rich",
                        pkl_path=onchain_paths.get("rich_pkl") or pkl_rich,
                        val_mode=outer.get("mode", "days"),
                        val_days=int(outer.get("days", 90)),
                        val_ratio=float(outer.get("ratio", 0.2)),
                        allowlist_path=allow_rich,
                        targets_override=union_targets,
                        cfg=embedded_params_rich,
                    )
                    embedded_rich_summary = emb_rich.summary
                    print(f"  [embedded] rich summary_rows={len(emb_rich.summary)}")
                except Exception as ee:  # pragma: no cover - defensive
                    print(f"  [embedded-warn] rich channel failed: {ee}")

            summary_base = None
            summary_rich = None

            # Step 1: base tree+perm
            if enable_base and pkl_base:
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
                targets_override=union_targets,
                embargo=perm_embargo,
                embargo_by_period=perm_embargo_by,
                purge=perm_purge,
                purge_by_period=perm_purge_by,
                n_estimators=perm_estimators,
                random_state=perm_seed,
            )
                print(" done")

            ran_rich = False
            if enable_rich and (pkl_rich or onchain_paths.get("rich_pkl")):
                print("  Step 2/4: rich tree+perm ...", end="", flush=True)
                summary_rich = run_tree_perm(
                    periods=periods,
                    val_mode=outer.get("mode", "days"),
                    val_days=int(outer.get("days", 90)),
                    val_ratio=float(outer.get("ratio", 0.2)),
                    topn_preview=3,
                    out_dir=tree_out_rich,
                    allowlist_path=allow_rich,
                    pkl_path=onchain_paths.get("rich_pkl") or pkl_rich,
                    time_perm=bool(perm_enabled),
                    perm_method=str(perm.get("method", "cyclic_shift")),
                    block_len=default_block_len,
                    block_len_by_period=block_len_map,
                    group_cols=group_cols,
                    repeats=perm_repeats,
                    targets_override=union_targets,
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
            if enable_rich and ran_rich:
                core_df = combine_channels(
                    base_dir=tree_out_base,
                    rich_dir=tree_out_rich,
                    weights_yaml=weights_yaml,
                    topk=int(agg.get("topk_core", 128)),
                    topk_per_pair=int(agg.get("topk_per_pair", 64)) or None,
                    min_appear_rate=float(agg.get("min_appear_rate", 0.5)) or None,
                    rich_quality_weights=rich_quality_weights if isinstance(rich_quality_weights, dict) else {},
                    min_appear_rate_era=float(min_appear_rate_era) if isinstance(min_appear_rate_era, (int, float)) else None,
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
                    min_appear_rate_era=float(min_appear_rate_era) if isinstance(min_appear_rate_era, (int, float)) else None,
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
                pkl_path=(pkl_rich if (enable_rich and pkl_rich) else pkl_base),
                val_mode=outer.get("mode", "days"),
                val_days=int(outer.get("days", 90)),
                val_ratio=float(outer.get("ratio", 0.2)),
                allowlist_path=export_path,
                targets_override=union_targets,
            )
            core_feats = core_df[core_df["keep"]]["feature"].astype(str).tolist()
            pool = core_feats.copy()
            method = (wrapper.get("method", "rfe") or "rfe").lower()
            if method == "ga":
                seeds_list = wrapper_ga_cfg.get("seeds") or []
                if isinstance(seeds_list, list) and len(seeds_list) > 1:
                    try:
                        seeds = [int(s) for s in seeds_list]
                    except Exception:
                        seeds = [42]
                    feats, score, det, freq = ga_search_multi(
                        pool,
                        ds.train,
                        ds.val,
                        ds.targets,
                        weights_yaml,
                        seeds=seeds,
                        pop_size=int(wrapper_ga_cfg.get("pop", 30)),
                        generations=int(wrapper_ga_cfg.get("gen", 15)),
                        cx_prob=float(wrapper_ga_cfg.get("cx", 0.7)),
                        mut_prob=float(wrapper_ga_cfg.get("mut", 0.1)),
                        sample_cap=int(wrapper_ga_cfg.get("sample_cap", 50000)),
                        multi_objective=ga_multi_objective,
                        mo_weights=ga_weights,
                    )
                    # save high-frequency loci (>60%)
                    freq_items = sorted([(k, v) for k, v in freq.items()], key=lambda x: x[1], reverse=True)
                    loci_dir = core_dir
                    os.makedirs(loci_dir, exist_ok=True)
                    loci_path = os.path.join(loci_dir, "ga_gene_frequency.csv")
                    import csv
                    with open(loci_path, "w", encoding="utf-8", newline="") as fcsv:
                        writer = csv.writer(fcsv)
                        writer.writerow(["feature", "freq"])
                        for k, v in freq_items:
                            writer.writerow([k, v])
                else:
                    from features.selection.optimize_subset import ga_search as _ga_single
                    feats, score, det = _ga_single(
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
            # forward compare: subset vs full features（可选）
            forward_cfg = finalize.get("forward_compare", {}) or {}
            if bool(forward_cfg.get("enabled", False)):
                try:
                    # subset features
                    sc_sub = _eval_once(combined_feats, ds.train, ds.val, ds.targets, weights_yaml)
                    # full features（使用当前 ds 的特征列）
                    sc_full = _eval_once(ds.features, ds.train, ds.val, ds.targets, weights_yaml)
                    summary_dict["forward_compare"] = {"subset_score": float(sc_sub), "full_score": float(sc_full)}
                    fwd_path = os.path.join(core_dir, "forward_eval.csv")
                    import csv
                    with open(fwd_path, "w", encoding="utf-8", newline="") as fcsv:
                        writer = csv.writer(fcsv)
                        writer.writerow(["variant", "score"])
                        writer.writerow(["subset", float(sc_sub)])
                        writer.writerow(["full", float(sc_full)])
                except Exception as fe:
                    summary_dict.setdefault("warnings", []).append(f"forward_compare failed: {fe}")
                channel_info: Dict[str, Any] = {}
                if res_base:
                    base_dir = Path(tree_out_root) / name / "base"
                    channel_info["base"] = {
                        "filter_kept": len(res_base.keep_features),
                        "filter_dropped": int(res_base.dropped.shape[0]) if isinstance(res_base.dropped, pd.DataFrame) else None,
                        "filter_allowlist": str(res_base.allowlist_path),
                        "filter_stats": str(Path(res_base.allowlist_path).parent / "filter_stats.csv"),
                        "ic_mi_csv": str(Path(res_base.allowlist_path).parent / "ic_mi.csv"),
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
                        "ic_mi_csv": str(Path(res_rich.allowlist_path).parent / "ic_mi.csv"),
                        "embedded_rows": int(embedded_rich_summary.shape[0]) if embedded_rich_summary is not None else None,
                        "embedded_summary": str(rich_dir / "stage2_embedded" / "summary.csv"),
                        "tree_perm_summary": str(rich_dir / "tree_perm" / "summary.csv"),
                    }
                if channel_info:
                    summary_dict["channels"] = channel_info
                # 记录关键阈值/种子/路径
                summary_dict["config_snapshot"] = {
                    "filter": {
                        "coverage_threshold": filter_params_default.get("coverage_threshold"),
                        "variance_threshold": filter_params_default.get("variance_threshold"),
                        "corr_threshold": filter_params_default.get("corr_threshold"),
                        "vif_threshold": filter_params_default.get("vif_threshold"),
                        "max_vif_iter": filter_params_default.get("max_vif_iter"),
                    },
                    "permutation": {
                        "enabled": bool(perm_enabled),
                        "block_len": default_block_len,
                        "block_len_by_period": block_len_map,
                        "repeats": perm_repeats,
                        "embargo": perm_embargo,
                        "purge": perm_purge,
                        "seed": perm_seed,
                    },
                    "aggregation": {
                        "topk_core": agg.get("topk_core"),
                        "topk_per_pair": agg.get("topk_per_pair"),
                        "min_appear_rate": agg.get("min_appear_rate"),
                        "min_appear_rate_era": min_appear_rate_era,
                    },
                    "wrapper": {
                        "method": method,
                        "ga": wrapper_ga_cfg,
                        "rfe": wrapper.get("rfe", {}),
                    },
                }
                # 剔除理由 Top-N（来自 filter dropped）
                try:
                    dropped_all = []
                    if res_base:
                        dropped_all.append(pd.read_csv(Path(res_base.allowlist_path).parent / "dropped.csv"))
                    if res_rich:
                        dropped_all.append(pd.read_csv(Path(res_rich.allowlist_path).parent / "dropped.csv"))
                    if dropped_all:
                        dcat = pd.concat(dropped_all, ignore_index=True)
                        top_reasons = dcat["reason"].value_counts().head(5).to_dict()
                        summary_dict["top_drop_reasons"] = {k: int(v) for k, v in top_reasons.items()}
                except Exception:
                    pass
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
