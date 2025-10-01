from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, RidgeCV, LassoCV
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.preprocessing import StandardScaler

from .common import DatasetSplit, is_classification_target, load_split, safe_numeric_copy

try:  # optional SHAP support
    import shap
except Exception:  # pragma: no cover - shap optional
    shap = None

# note: do not globally silence convergence warnings; handle per-fit with fallback


@dataclass
class EmbeddedParams:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    random_state: int = 42
    sample_cap: int = 60000
    boruta_iterations: int = 20
    linear_cv: int = 5
    linear_max_iter: int = 2000
    tree_max_features: Optional[float] = None
    n_jobs: int = -1
    # GPU/backend controls
    tree_backend: str = "xgboost"  # xgboost | catboost | sklearn
    use_gpu: bool = True
    # Optional VSN integration
    use_vsn: bool = True
    vsn_csv: Optional[str] = None
    # Weighted aggregation of evidence
    w_tree: float = 1.0
    w_linear: float = 1.0
    w_boruta: float = 1.0
    w_shap: float = 1.0
    w_vsn: float = 1.0


@dataclass
class EmbeddedResult:
    summary: pd.DataFrame
    raw_scores: pd.DataFrame
    allowlist_path: Optional[Path]


def _output_dir(expert_name: str, channel: str) -> Path:
    return Path("reports/feature_evidence") / expert_name / channel / "stage2_embedded"


def _prepare_xy(df: pd.DataFrame, features: List[str], target: str, is_cls: bool) -> tuple[np.ndarray, np.ndarray, List[str]]:
    cols = [c for c in features if c in df.columns]
    X = safe_numeric_copy(df[cols])
    med = X.median(axis=0, numeric_only=True)
    X = X.fillna(med).replace([np.inf, -np.inf], 0)
    if is_cls:
        y = df[target].fillna(0).clip(0, 1).astype(int)
    else:
        y = df[target].astype(float)
    return X.to_numpy(dtype=float), y.to_numpy(), cols


def _get_tree_model(
    *,
    is_cls: bool,
    params: EmbeddedParams,
    n_estimators: Optional[int] = None,
    random_state_override: Optional[int] = None,
):
    backend = (params.tree_backend or "xgboost").lower()
    use_gpu = bool(params.use_gpu)
    n_est = int(n_estimators if n_estimators is not None else params.n_estimators)
    max_depth_val = int(params.max_depth) if params.max_depth is not None else None
    max_features = params.tree_max_features
    n_jobs = int(params.n_jobs)
    seed = int(random_state_override if random_state_override is not None else params.random_state)

    if backend == "xgboost":
        try:
            import xgboost as xgb  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            xgb = None
        if xgb is not None:
            common = dict(
                n_estimators=n_est,
                max_depth=(max_depth_val if max_depth_val is not None else 6),
                subsample=1.0,
                colsample_bytree=float(max_features) if (isinstance(max_features, (int, float)) and max_features) else 1.0,
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=0,
            )
            if use_gpu:
                try:
                    params_gpu = dict(common)
                    params_gpu.update(tree_method="hist", predictor="auto", device="cuda")
                    if is_cls:
                        return xgb.XGBClassifier(**params_gpu, eval_metric="logloss")
                    else:
                        return xgb.XGBRegressor(**params_gpu, eval_metric="rmse")
                except TypeError:
                    params_legacy = dict(common)
                    params_legacy.update(tree_method="gpu_hist", predictor="gpu_predictor")
                    if is_cls:
                        return xgb.XGBClassifier(**params_legacy, eval_metric="logloss")
                    else:
                        return xgb.XGBRegressor(**params_legacy, eval_metric="rmse")
            params_cpu = dict(common)
            params_cpu.update(tree_method="hist", predictor="auto")
            if is_cls:
                return xgb.XGBClassifier(**params_cpu, eval_metric="logloss")
            else:
                return xgb.XGBRegressor(**params_cpu, eval_metric="rmse")
    if backend == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            CatBoostClassifier = None
            CatBoostRegressor = None
        if CatBoostClassifier is not None and CatBoostRegressor is not None:
            common_cb = dict(
                iterations=n_est,
                depth=(params.max_depth if params.max_depth is not None else 6),
                random_seed=seed,
                task_type=("GPU" if use_gpu else "CPU"),
                verbose=False,
                thread_count=n_jobs if isinstance(n_jobs, int) else None,
            )
            return CatBoostClassifier(**common_cb) if is_cls else CatBoostRegressor(**common_cb)

    # fallback to sklearn RF (CPU)
    if is_cls:
        model = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=params.max_depth,
            max_features=max_features,
            n_jobs=n_jobs,
            class_weight="balanced_subsample",
            random_state=seed,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=params.max_depth,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=seed,
        )
    return model


def _fit_tree_importance(
    X: np.ndarray,
    y: np.ndarray,
    cols: List[str],
    is_cls: bool,
    params: EmbeddedParams,
) -> tuple[pd.Series, pd.Series]:
    if X.size == 0:
        empty = pd.Series(dtype=float)
        return empty, empty
    model = _get_tree_model(is_cls=is_cls, params=params)
    model.fit(X, y)
    # feature_importances_
    try:
        importances = pd.Series(getattr(model, "feature_importances_"), index=cols, dtype=float)
    except Exception:
        try:
            # CatBoost fallback
            imp = model.get_feature_importance()
            importances = pd.Series(imp, index=cols, dtype=float)
        except Exception:
            importances = pd.Series(np.zeros(len(cols), dtype=float), index=cols)
    shap_series = pd.Series(np.nan, index=cols, dtype=float)
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_matrix = shap_values[1] if is_cls and len(shap_values) > 1 else shap_values[0]
            else:
                shap_matrix = shap_values
            shap_abs = np.abs(np.asarray(shap_matrix)).mean(axis=0)
            shap_series = pd.Series(shap_abs, index=cols, dtype=float)
        except Exception:
            pass
    return importances, shap_series


def _run_boruta_like(X: np.ndarray, y: np.ndarray, cols: List[str], is_cls: bool, params: EmbeddedParams) -> pd.Series:
    if X.size == 0:
        return pd.Series(dtype=float)
    rng = np.random.RandomState(params.random_state)
    keep_scores = np.zeros(len(cols), dtype=float)
    for it in range(max(1, params.boruta_iterations)):
        shadow = X.copy()
        for j in range(shadow.shape[1]):
            rng.shuffle(shadow[:, j])
        X_aug = np.concatenate([X, shadow], axis=1)
        model = _get_tree_model(
            is_cls=is_cls,
            params=params,
            n_estimators=200,
            random_state_override=params.random_state + it,
        )
        model.fit(X_aug, y)
        try:
            importances = getattr(model, "feature_importances_")
        except Exception:
            try:
                importances = model.get_feature_importance()
            except Exception:
                importances = np.zeros(X_aug.shape[1], dtype=float)
        shadow_max = float(importances[len(cols) :].max(initial=0.0))
        keep_scores += (importances[: len(cols)] > shadow_max).astype(float)
    keep_scores /= max(1, params.boruta_iterations)
    return pd.Series(keep_scores, index=cols, dtype=float)


def _run_linear_path(X: np.ndarray, y: np.ndarray, cols: List[str], is_cls: bool, params: EmbeddedParams) -> pd.Series:
    if X.size == 0:
        return pd.Series(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if is_cls:
        coef = None
        # Try robust L1 with saga first
        try:
            model = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0],
                cv=params.linear_cv,
                penalty="l1",
                solver="saga",
                max_iter=params.linear_max_iter,
                n_jobs=1,
                multi_class="auto",
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                model.fit(X_scaled, y)
            coef_abs = np.abs(model.coef_).mean(axis=0)
            # clip extreme coefficients at 99th percentile for stability
            cap = float(np.percentile(coef_abs, 99)) if np.isfinite(coef_abs).any() else float("inf")
            coef = np.clip(coef_abs, 0.0, cap)
        except Exception:
            pass
        # Fallback to stable L2 with lbfgs
        if coef is None or not np.isfinite(np.asarray(coef)).all():
            try:
                model = LogisticRegressionCV(
                    Cs=[0.01, 0.1, 1.0, 10.0],
                    cv=params.linear_cv,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=params.linear_max_iter,
                    n_jobs=1,
                    multi_class="auto",
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    model.fit(X_scaled, y)
                coef_abs = np.abs(model.coef_).mean(axis=0)
                cap = float(np.percentile(coef_abs, 99)) if np.isfinite(coef_abs).any() else float("inf")
                coef = np.clip(coef_abs, 0.0, cap)
            except Exception:
                coef = np.zeros(X.shape[1], dtype=float)
    else:
        coef = None
        # Primary: ElasticNetCV
        try:
            model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                cv=params.linear_cv,
                max_iter=params.linear_max_iter,
                n_jobs=params.n_jobs,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                model.fit(X_scaled, y)
            coef_abs = np.abs(model.coef_)
            cap = float(np.percentile(coef_abs, 99)) if np.isfinite(coef_abs).any() else float("inf")
            coef = np.clip(coef_abs, 0.0, cap)
        except Exception:
            pass
        # Fallback 1: RidgeCV (stable closed-form-like)
        if coef is None or not np.isfinite(np.asarray(coef)).all():
            try:
                ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
                ridge.fit(X_scaled, y)
                coef_abs = np.abs(getattr(ridge, "coef_", np.zeros(X.shape[1])))
                cap = float(np.percentile(coef_abs, 99)) if np.isfinite(coef_abs).any() else float("inf")
                coef = np.clip(coef_abs, 0.0, cap)
            except Exception:
                pass
        # Fallback 2: LassoCV as sparsity alternative
        if coef is None or not np.isfinite(np.asarray(coef)).all():
            try:
                lasso = LassoCV(max_iter=params.linear_max_iter, n_jobs=params.n_jobs)
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    lasso.fit(X_scaled, y)
                coef_abs = np.abs(getattr(lasso, "coef_", np.zeros(X.shape[1])))
                cap = float(np.percentile(coef_abs, 99)) if np.isfinite(coef_abs).any() else float("inf")
                coef = np.clip(coef_abs, 0.0, cap)
            except Exception:
                coef = np.zeros(X.shape[1], dtype=float)
    return pd.Series(coef, index=cols, dtype=float)


def _aggregate_scores(rows: List[pd.DataFrame], *, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["feature", "period", "target", "tree_importance", "boruta_keep", "linear_coef", "shap_value", "tft_vsn_importance", "score_embedded"])
    df = pd.concat(rows, ignore_index=True)
    groups = df.groupby(["period", "target"], sort=False)
    n = groups["feature"].transform("count").astype(float).clip(lower=1.0)
    df["tree_rank"] = groups["tree_importance"].rank(method="average", ascending=False)
    df["linear_rank"] = groups["linear_coef"].rank(method="average", ascending=False)
    df["boruta_score"] = df["boruta_keep"].fillna(0.0)
    tree_score = 1.0 - (df["tree_rank"] - 1.0) / np.where(n > 1.0, n - 1.0, 1.0)
    linear_score = 1.0 - (df["linear_rank"] - 1.0) / np.where(n > 1.0, n - 1.0, 1.0)
    shap_score = np.zeros(len(df), dtype=float)
    if "shap_value" in df.columns:
        shap_rank = groups["shap_value"].rank(method="average", ascending=False)
        shap_score = 1.0 - (shap_rank - 1.0) / np.where(n > 1.0, n - 1.0, 1.0)
    vsn_score = np.zeros(len(df), dtype=float)
    if "tft_vsn_importance" in df.columns:
        vsn_rank = groups["tft_vsn_importance"].rank(method="average", ascending=False)
        vsn_score = 1.0 - (vsn_rank - 1.0) / np.where(n > 1.0, n - 1.0, 1.0)
    # weighted aggregation
    w = dict(weights or {})
    w_tree = float(w.get("tree", 1.0))
    w_linear = float(w.get("linear", 1.0))
    w_boruta = float(w.get("boruta", 1.0))
    w_shap = float(w.get("shap", 1.0))
    w_vsn = float(w.get("vsn", 1.0))
    num = (
        w_tree * tree_score
        + w_linear * linear_score
        + w_boruta * df["boruta_score"].to_numpy()
        + w_shap * shap_score
        + w_vsn * vsn_score
    )
    denom = np.maximum(1e-12, (w_tree * (tree_score > -1)).astype(float) + (w_linear * (linear_score > -1)).astype(float) + (w_boruta * (df["boruta_score"].to_numpy() >= 0)).astype(float) + (w_shap * (shap_score >= 0)).astype(float) + (w_vsn * (vsn_score >= 0)).astype(float))
    df["score_embedded"] = np.nan_to_num(num / denom)
    return df


def run_embedded_stage(
    *,
    pkl_path: str,
    expert_name: str,
    channel: str,
    val_mode: str,
    val_days: int,
    val_ratio: float,
    allowlist_path: Optional[str],
    params: Optional[EmbeddedParams] = None,
    targets_override: Optional[List[str]] = None,
    periods_override: Optional[List[str]] = None,
) -> EmbeddedResult:
    out_dir = _output_dir(expert_name, channel)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params or EmbeddedParams()
    print(f"[embedded] expert={expert_name} channel={channel} | backend={params.tree_backend} gpu={params.use_gpu}", flush=True)
    ds = load_split(
        pkl_path=pkl_path,
        val_mode=val_mode,
        val_days=val_days,
        val_ratio=val_ratio,
        allowlist_path=allowlist_path,
        targets_override=targets_override,
    )
    rows: List[pd.DataFrame] = []
    periods = periods_override or ds.periods
    features = ds.features

    # optional: load VSN (variable selection) importance from TFT checkpoint exported CSV
    vsn_scores_map: Dict[str, float] = {}
    if params.use_vsn:
        cand_paths = []
        if params.vsn_csv:
            cand_paths.append(params.vsn_csv)
        # conventional locations
        out_dir = _output_dir(expert_name, channel)
        cand_paths.append(str(out_dir.parent / "tft_gating.csv"))
        cand_paths.append(str(out_dir / "tft_gating.csv"))
        for pth in cand_paths:
            try:
                if pth and Path(pth).exists():
                    df_vsn = pd.read_csv(pth)
                    if {"feature", "score"}.issubset(df_vsn.columns):
                        vsn_scores_map = {str(r["feature"]): float(r["score"]) for _, r in df_vsn.iterrows()}
                        break
            except Exception:
                pass

    for period in periods:
        tr_df = ds.train[ds.train["period"].astype(str) == str(period)]
        if tr_df.empty:
            continue
        if params.sample_cap and len(tr_df) > params.sample_cap:
            tr_df = tr_df.sample(params.sample_cap, random_state=params.random_state)
        print(f"[embedded] period={period} | rows={len(tr_df)} targets={len(ds.targets)} (processing all periods from dataset)", flush=True)
        for target in ds.targets:
            if target not in tr_df.columns:
                continue
            is_cls = is_classification_target(target)
            X, y, cols = _prepare_xy(tr_df, features, target, is_cls)
            if X.size == 0:
                continue
            print(f"[embedded]  target={target} | steps=[tree,boruta,linear]", flush=True)
            tree_imp, shap_val = _fit_tree_importance(X, y, cols, is_cls, params)
            boruta_score = _run_boruta_like(X, y, cols, is_cls, params)
            linear_coef = _run_linear_path(X, y, cols, is_cls, params)
            vsn_vals = pd.Series({c: float(vsn_scores_map.get(c, np.nan)) for c in cols}, dtype=float)
            sub = pd.DataFrame({
                "feature": cols,
                "period": str(period),
                "target": target,
                "tree_importance": tree_imp.reindex(cols).fillna(0.0).to_numpy(),
                "boruta_keep": boruta_score.reindex(cols).fillna(0.0).to_numpy(),
                "linear_coef": linear_coef.reindex(cols).fillna(0.0).to_numpy(),
                "shap_value": shap_val.reindex(cols).fillna(0.0).to_numpy(),
                "tft_vsn_importance": vsn_vals.reindex(cols).fillna(0.0).to_numpy(),
            })
            rows.append(sub)
    weight_map = {
        "tree": float(params.w_tree),
        "linear": float(params.w_linear),
        "boruta": float(params.w_boruta),
        "shap": float(params.w_shap),
        "vsn": float(params.w_vsn),
    }
    summary = _aggregate_scores(rows, weights=weight_map)
    print(f"[embedded] done | rows={len(summary)} out_dir={out_dir}", flush=True)

    raw_scores = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["feature", "period", "target", "tree_importance", "boruta_keep", "linear_coef", "shap_value", "tft_vsn_importance"])
    if not raw_scores.empty:
        raw_scores.to_csv(out_dir / "raw_scores.csv", index=False)
    if not summary.empty:
        summary.to_csv(out_dir / "summary.csv", index=False)
        # evidence 明细（便于文档汇总）
        summary.to_csv(out_dir / "embedded_evidence.csv", index=False)

    allow_path = None
    if not summary.empty:
        keep = summary.groupby("feature")["score_embedded"].mean().sort_values(ascending=False)
        allow_path = out_dir / "allowlist_embedded.txt"
        with allow_path.open("w", encoding="utf-8") as fh:
            for feat in keep.index:
                fh.write(f"{feat}\n")

    return EmbeddedResult(summary=summary, raw_scores=raw_scores, allowlist_path=allow_path)


def run_embedded_for_channel(
    expert_name: str,
    channel: str,
    pkl_path: str,
    val_mode: str,
    val_days: int,
    val_ratio: float,
    allowlist_path: Optional[str],
    targets_override: Optional[List[str]] = None,
    cfg: Dict[str, object] | None = None,
    periods_override: Optional[List[str]] = None,
) -> EmbeddedResult:
    params = EmbeddedParams()
    if cfg:
        params = EmbeddedParams(
            n_estimators=int(cfg.get("n_estimators", params.n_estimators)),
            max_depth=cfg.get("max_depth", params.max_depth),
            random_state=int(cfg.get("random_state", params.random_state)),
            sample_cap=int(cfg.get("sample_cap", params.sample_cap)),
            boruta_iterations=int(cfg.get("boruta_iterations", params.boruta_iterations)),
            linear_cv=int(cfg.get("linear_cv", params.linear_cv)),
            linear_max_iter=int(cfg.get("linear_max_iter", params.linear_max_iter)),
            tree_max_features=cfg.get("tree_max_features", params.tree_max_features),
            n_jobs=int(cfg.get("n_jobs", params.n_jobs)),
            tree_backend=str(cfg.get("tree_backend", params.tree_backend)),
            use_gpu=bool(cfg.get("use_gpu", params.use_gpu)),
            w_tree=float(cfg.get("w_tree", params.w_tree)),
            w_linear=float(cfg.get("w_linear", params.w_linear)),
            w_boruta=float(cfg.get("w_boruta", params.w_boruta)),
            w_shap=float(cfg.get("w_shap", params.w_shap)),
            w_vsn=float(cfg.get("w_vsn", params.w_vsn)),
        )
    return run_embedded_stage(
        pkl_path=pkl_path,
        expert_name=expert_name,
        channel=channel,
        val_mode=val_mode,
        val_days=val_days,
        val_ratio=val_ratio,
        allowlist_path=allowlist_path,
        targets_override=targets_override,
        periods_override=periods_override,
        params=params,
    )
