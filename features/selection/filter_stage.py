from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .common import DatasetSplit, load_split, safe_numeric_copy


@dataclass
class FilterParams:
    coverage_threshold: float = 0.4
    variance_threshold: float = 1e-8
    corr_threshold: float = 0.97
    vif_threshold: float = 20.0
    max_vif_iter: int = 20
    suffix_dedup: Iterable[str] = ("_zn", "_mm")
    ic_targets: Iterable[str] | None = None
    mi_targets: Iterable[str] | None = None


@dataclass
class FilterResult:
    keep_features: List[str]
    dropped: pd.DataFrame
    summary: pd.DataFrame
    icmi: pd.DataFrame
    allowlist_path: Path


def _compose_output_dir(expert_name: str, channel: str) -> Path:
    return Path("reports/feature_evidence") / expert_name / channel / "stage1_filter"


def _ensure_float(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.loc[:, cols].copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _compute_vif(frame: pd.DataFrame) -> pd.Series:
    import numpy.linalg as npl

    X = frame.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    vif_values = []
    for i in range(X.shape[1]):
        y = X[:, i]
        X_i = np.delete(X, i, axis=1)
        XtX = X_i.T @ X_i
        try:
            beta = npl.solve(XtX, X_i.T @ y)
        except npl.LinAlgError:
            vif_values.append(np.inf)
            continue
        y_hat = X_i @ beta
        resid = y - y_hat
        ssr = float(np.sum(resid ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ssr / max(1e-12, sst)
        vif = 1.0 / max(1e-12, 1.0 - r2)
        vif_values.append(vif)
    return pd.Series(vif_values, index=frame.columns, dtype=float)


def _cluster_by_corr(df: pd.DataFrame, threshold: float) -> Dict[str, List[str]]:
    corr = df.corr(method="spearman").abs()
    corr = corr.fillna(0.0)
    features = list(corr.columns)
    parent = {f: f for f in features}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rx < ry:
            parent[ry] = rx
        else:
            parent[rx] = ry

    for i, fi in enumerate(features):
        for fj in features[i + 1 :]:
            if corr.at[fi, fj] >= threshold:
                union(fi, fj)

    clusters: Dict[str, List[str]] = {}
    for f in features:
        root = find(f)
        clusters.setdefault(root, []).append(f)
    return clusters


def _choose_representatives(
    clusters: Dict[str, List[str]],
    stats: pd.DataFrame,
    suffixes: Iterable[str],
) -> Tuple[List[str], Dict[str, str]]:
    keep: List[str] = []
    mapping: Dict[str, str] = {}
    suffixes = tuple(suffixes)

    def _base_name(name: str) -> str:
        for suf in suffixes:
            if name.endswith(suf):
                return name[: -len(suf)]
        return name

    seen_base: Dict[str, str] = {}
    for root, members in clusters.items():
        subset = stats.loc[members].sort_values(
            ["coverage", "variance"], ascending=[False, False]
        )
        rep = subset.index[0]
        base = _base_name(rep)
        if base in seen_base:
            rep = seen_base[base]
        else:
            seen_base[base] = rep
        keep.append(rep)
        for m in members:
            mapping[m] = rep
    return keep, mapping


def _ic_mi_scores(
    ds: DatasetSplit,
    params: FilterParams,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    data = pd.concat([ds.train, ds.val], ignore_index=True)
    X = safe_numeric_copy(data[stats.index])
    med = X.median(axis=0, numeric_only=True)
    X = X.fillna(med)
    for tgt in params.ic_targets or []:
        if tgt in data.columns:
            y = data[tgt].astype(float)
            for feat in stats.index:
                rows.append(
                    {
                        "feature": feat,
                        "target": tgt,
                        "metric": "ic",
                        "value": float(X[feat].corr(y, method="spearman")),
                    }
                )
    for tgt in params.mi_targets or []:
        if tgt in data.columns:
            y = data[tgt].fillna(0).clip(0, 1).astype(int)
            mi = mutual_info_classif(X, y, random_state=42)
            for feat, val in zip(stats.index, mi):
                rows.append(
                    {
                        "feature": feat,
                        "target": tgt,
                        "metric": "mi",
                        "value": float(val),
                    }
                )
    return pd.DataFrame(rows)


def run_filter_stage(
    *,
    pkl_path: str,
    expert_name: str,
    channel: str,
    val_mode: str,
    val_days: int,
    val_ratio: float,
    allowlist_path: str | None,
    params: FilterParams,
) -> FilterResult:
    out_dir = _compose_output_dir(expert_name, channel)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_split(
        pkl_path=pkl_path,
        val_mode=val_mode,
        val_days=val_days,
        val_ratio=val_ratio,
        allowlist_path=allowlist_path,
    )
    print(f"[filter] expert={expert_name} channel={channel} | periods={len(ds.periods)} targets={len(ds.targets)}", flush=True)
    full = pd.concat([ds.train, ds.val], ignore_index=True)
    features = ds.features
    numeric_df = _ensure_float(full, features)
    print(f"[filter] start | total_features={len(features)} rows={len(numeric_df)}", flush=True)

    coverage = 1.0 - numeric_df.isna().mean()
    variance = numeric_df.var(axis=0, numeric_only=True)
    stats = pd.DataFrame({"coverage": coverage, "variance": variance})
    stats.index.name = "feature"

    dropped_rows = []
    keep_mask = coverage >= params.coverage_threshold
    dropped_rows.extend(
        [
            (f, "coverage", coverage[f])
            for f in coverage.index[~keep_mask]
        ]
    )
    numeric_df = numeric_df.loc[:, keep_mask.values]
    stats = stats.loc[keep_mask]
    print(f"[filter] step1 coverage>= {params.coverage_threshold} | kept={numeric_df.shape[1]} dropped={len(features)-numeric_df.shape[1]}", flush=True)

    low_var_mask = stats["variance"] >= params.variance_threshold
    dropped_rows.extend(
        [
            (f, "variance", stats.loc[f, "variance"])
            for f in stats.index[~low_var_mask]
        ]
    )
    numeric_df = numeric_df.loc[:, low_var_mask.values]
    stats = stats.loc[low_var_mask]
    print(f"[filter] step2 variance>= {params.variance_threshold} | kept={numeric_df.shape[1]}", flush=True)

    if numeric_df.shape[1] == 0:
        dropped_df = pd.DataFrame(dropped_rows, columns=["feature", "reason", "value"])
        stats.to_csv(out_dir / "filter_stats.csv")
        dropped_df.to_csv(out_dir / "dropped.csv", index=False)
        allow = out_dir / "allowlist.txt"
        allow.write_text("", encoding="utf-8")
        return FilterResult([], dropped_df, stats, pd.DataFrame(), allow)

    corr_clusters = _cluster_by_corr(numeric_df, params.corr_threshold)
    keep_features, cluster_map = _choose_representatives(
        corr_clusters, stats, params.suffix_dedup
    )
    for feat, rep in cluster_map.items():
        if feat != rep:
            dropped_rows.append((feat, "corr_cluster", rep))
    print(f"[filter] step3 corr>= {params.corr_threshold} | clusters={len(corr_clusters)} reps={len(keep_features)}", flush=True)

    vif_frame = numeric_df[keep_features].copy()
    vif_iter = 0
    while vif_frame.shape[1] > 1 and vif_iter < params.max_vif_iter:
        vif_iter += 1
        vif_values = _compute_vif(vif_frame)
        max_vif = vif_values.max()
        if max_vif <= params.vif_threshold:
            break
        worst = vif_values.sort_values(ascending=False).index[0]
        vif_frame = vif_frame.drop(columns=[worst])
        keep_features.remove(worst)
        dropped_rows.append((worst, "vif", float(max_vif)))
        print(f"[filter] step4 VIF iter={vif_iter}/{params.max_vif_iter} | drop={worst} max_vif={float(max_vif):.3f} kept={len(keep_features)}", flush=True)

    icmi = _ic_mi_scores(ds, params, stats.loc[keep_features])
    print(f"[filter] step5 IC/MI | rows={len(icmi)}", flush=True)

    stats.loc[:, "keep"] = stats.index.isin(keep_features)
    stats.loc[:, "cluster_rep"] = [cluster_map.get(f, f) for f in stats.index]

    dropped_df = pd.DataFrame(dropped_rows, columns=["feature", "reason", "value"])
    stats.to_csv(out_dir / "filter_stats.csv")
    dropped_df.to_csv(out_dir / "dropped.csv", index=False)
    if not icmi.empty:
        icmi.to_csv(out_dir / "ic_mi.csv", index=False)

    allow_path = out_dir / "allowlist.txt"
    with allow_path.open("w", encoding="utf-8") as fh:
        for feat in keep_features:
            fh.write(f"{feat}\n")
    print(f"[filter] done | keep={len(keep_features)} allowlist={allow_path}", flush=True)

    return FilterResult(keep_features, dropped_df, stats, icmi, allow_path)


def run_filter_for_channel(
    expert_name: str,
    channel: str,
    pkl_path: str,
    val_mode: str,
    val_days: int,
    val_ratio: float,
    allowlist_path: str | None,
    params_dict: Dict[str, object] | None,
) -> FilterResult:
    params = FilterParams()
    if params_dict:
        params = FilterParams(
            coverage_threshold=float(params_dict.get("coverage_threshold", params.coverage_threshold)),
            variance_threshold=float(params_dict.get("variance_threshold", params.variance_threshold)),
            corr_threshold=float(params_dict.get("corr_threshold", params.corr_threshold)),
            vif_threshold=float(params_dict.get("vif_threshold", params.vif_threshold)),
            max_vif_iter=int(params_dict.get("max_vif_iter", params.max_vif_iter)),
            suffix_dedup=tuple(params_dict.get("suffix_dedup", params.suffix_dedup)),
            ic_targets=params_dict.get("ic_targets"),
            mi_targets=params_dict.get("mi_targets"),
        )
    return run_filter_stage(
        pkl_path=pkl_path,
        expert_name=expert_name,
        channel=channel,
        val_mode=val_mode,
        val_days=val_days,
        val_ratio=val_ratio,
        allowlist_path=allowlist_path,
        params=params,
    )
