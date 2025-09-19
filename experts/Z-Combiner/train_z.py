
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from metrics.calibration import compute_ece


DEFAULT_FEATURE_PREFIXES = [
    "score__",
    "uncertainty__",
    "realized_vol",
    "ewma_vol",
    "volume_change",
    "funding_rate_slope",
    "open_interest_slope",
    "momentum_fast",
    "momentum_slow",
    "atr_slope",
    "structural_gap",
]

def _clip_probabilities(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 1e-6, 1 - 1e-6)


def _select_features(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    return sorted({col for col in df.columns if any(col.startswith(pref) for pref in prefixes)})


def _time_based_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values('time_idx')
    unique_times = df['time_idx'].unique()
    if len(unique_times) < 3:
        return df, df, df
    train_cut_idx = max(1, int(len(unique_times) * train_ratio))
    val_cut_idx = max(train_cut_idx + 1, int(len(unique_times) * (train_ratio + val_ratio)))
    train_cut = unique_times[train_cut_idx - 1]
    val_cut = unique_times[min(val_cut_idx, len(unique_times) - 1)]
    train_df = df[df['time_idx'] <= train_cut]
    val_df = df[(df['time_idx'] > train_cut) & (df['time_idx'] <= val_cut)]
    test_df = df[df['time_idx'] > val_cut]
    if val_df.empty:
        val_df = train_df.sample(frac=0.2, random_state=42)
        train_df = df.drop(val_df.index)
    if test_df.empty:
        test_df = val_df.copy()
    return train_df, val_df, test_df


def _baseline_equal_weight(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    score_cols = [c for c in feature_cols if c.startswith('score__')]
    selected = score_cols or feature_cols
    return df[selected].mean(axis=1).to_numpy()


def _best_single_expert(val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], val_target: np.ndarray, task: str) -> np.ndarray:
    score_cols = [c for c in feature_cols if c.startswith('score__')]
    selected = score_cols or feature_cols
    best_col = selected[0]
    if task == 'classification':
        best_score = -math.inf
        for col in selected:
            score = average_precision_score(val_target, val_df[col].to_numpy())
            if score > best_score:
                best_score = score
                best_col = col
    else:
        best_score = math.inf
        for col in selected:
            score = mean_squared_error(val_target, val_df[col].to_numpy(), squared=False)
            if score < best_score:
                best_score = score
                best_col = col
    return _clip_probabilities(test_df[best_col].to_numpy()) if task == 'classification' else test_df[best_col].to_numpy()


def _load_config(path: Path) -> dict:
    import yaml

    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def _prepare_features(df: pd.DataFrame, prefixes: List[str], target: str) -> tuple[List[str], pd.DataFrame]:
    feature_cols = _select_features(df, prefixes)
    if not feature_cols:
        raise ValueError('No features selected for Z-combiner training. Check feature prefixes.')
    cols = ['symbol', 'period', 'time_idx', target] + feature_cols
    extra_cols = [c for c in ('schema_ver', 'data_ver', 'expert_ver', 'train_window_id') if c in df.columns]
    cols.extend(extra_cols)
    subset = df[cols].dropna(subset=[target])
    return feature_cols, subset


def train_classification(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_regression(X_train: np.ndarray, y_train: np.ndarray, hidden_units: int = 64) -> Pipeline:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(hidden_units,), max_iter=300, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_classification(model: Pipeline, X: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float, float]:
    probs = model.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y_true, probs)
    ece = compute_ece(torch.from_numpy(probs), torch.from_numpy(y_true.astype(np.float32)))
    return _clip_probabilities(probs), float(pr_auc), float(ece)


def evaluate_regression(model: Pipeline, X: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float]:
    preds = model.predict(X)
    rmse = mean_squared_error(y_true, preds, squared=False)
    return preds, float(rmse)


def run_training(config_path: Path) -> None:
    cfg = _load_config(config_path)
    data_path = Path(cfg.get('data_path', 'datasets/z_train.parquet'))
    task_type = cfg.get('task_type', 'classification').lower()
    target_column = cfg.get('target_column', 'target_binarytrend')
    feature_prefixes = cfg.get('feature_prefixes', DEFAULT_FEATURE_PREFIXES)
    train_ratio = float(cfg.get('train_ratio', 0.6))
    val_ratio = float(cfg.get('val_ratio', 0.2))
    log_dir = Path(cfg.get('log_dir', 'lightning_logs/experts/Z-Combiner'))
    checkpoints_dir = Path(cfg.get('checkpoints_dir', 'checkpoints/Z-Combiner'))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    feature_cols, subset = _prepare_features(df, feature_prefixes, target_column)
    train_df, val_df, test_df = _time_based_split(subset, train_ratio, val_ratio)

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[target_column].to_numpy()
    X_val = val_df[feature_cols].to_numpy()
    y_val = val_df[target_column].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df[target_column].to_numpy()

    results = {}

    if task_type == 'classification':
        model = train_classification(X_train, y_train)
        _, pr_auc_val, ece_val = evaluate_classification(model, X_val, y_val)
        probs_test, pr_auc_test, ece_test = evaluate_classification(model, X_test, y_test)
        results['model'] = {
            'pr_auc_val': pr_auc_val,
            'ece_val': ece_val,
            'pr_auc_test': pr_auc_test,
            'ece_test': ece_test,
        }
        baseline_eq_raw = _baseline_equal_weight(test_df, feature_cols)
        baseline_eq = _clip_probabilities(baseline_eq_raw) if task_type == 'classification' else baseline_eq_raw
        baseline_best = _clip_probabilities(_best_single_expert(val_df, test_df, feature_cols, y_val, 'classification'))
        results['baseline_equal_weight'] = {
            'pr_auc': float(average_precision_score(y_test, baseline_eq)),
            'ece': float(compute_ece(torch.from_numpy(baseline_eq), torch.from_numpy(y_test.astype(np.float32)))),
        }
        results['baseline_best_expert'] = {
            'pr_auc': float(average_precision_score(y_test, baseline_best)),
            'ece': float(compute_ece(torch.from_numpy(baseline_best), torch.from_numpy(y_test.astype(np.float32)))),
        }
    else:
        model = train_regression(X_train, y_train)
        _, rmse_val = evaluate_regression(model, X_val, y_val)
        preds_test, rmse_test = evaluate_regression(model, X_test, y_test)
        results['model'] = {
            'rmse_val': rmse_val,
            'rmse_test': rmse_test,
        }
        baseline_eq_raw = _baseline_equal_weight(test_df, feature_cols)
        baseline_eq = _clip_probabilities(baseline_eq_raw) if task_type == 'classification' else baseline_eq_raw
        baseline_best = _best_single_expert(val_df, test_df, feature_cols, y_val, 'regression')
        results['baseline_equal_weight'] = {
            'rmse': float(mean_squared_error(y_test, baseline_eq, squared=False))
        }
        results['baseline_best_expert'] = {
            'rmse': float(mean_squared_error(y_test, baseline_best, squared=False))
        }

    metrics_path = log_dir / f"metrics_{target_column}.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    import joblib
    model_path = checkpoints_dir / f"z_combiner_{target_column}.joblib"
    joblib.dump(model, model_path)

    print(json.dumps(results, indent=2))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train Z-Combiner stacking model')
    parser.add_argument('--config', type=str, required=True, help='Path to Z-Combiner model_config.yaml')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    run_training(Path(args.config))


if __name__ == '__main__':
    main()
