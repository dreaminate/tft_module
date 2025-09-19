from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd

from features.regime_core import compute_regime_core_features


def _load_predictions(pred_root: Path) -> pd.DataFrame:
    files: List[Path] = sorted(pred_root.rglob("predictions_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No prediction parquet files found under {pred_root}")
    frames = []
    for fp in files:
        df = pd.read_parquet(fp)
        if df.empty:
            continue
        if 'expert' not in df.columns:
            raise ValueError(f"prediction file {fp} lacks 'expert' column")
        frames.append(df)
    if not frames:
        raise RuntimeError("All prediction files were empty")
    return pd.concat(frames, ignore_index=True)


def _pivot_predictions(pred_df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    pivot = pred_df.pivot_table(
        index=['symbol', 'period', 'time_idx'],
        columns=['expert', 'target'],
        values=value_col,
        aggfunc='mean',
    )
    pivot.columns = [f"{prefix}__{expert}__{target}" for expert, target in pivot.columns]
    return pivot.reset_index()


def build_oof_dataset(
    predictions_root: Path,
    data_path: Path,
    output_path: Path,
    target_columns: List[str] | None = None,
    regime_enabled: bool = True,
) -> Path:
    pred_df = _load_predictions(predictions_root)

    required_meta = ['schema_ver', 'data_ver', 'expert_ver', 'train_window_id']
    for col in required_meta:
        if col not in pred_df.columns:
            raise ValueError(f"Prediction files missing column '{col}'")
    meta_values = pred_df[required_meta].drop_duplicates()
    if len(meta_values) > 1:
        raise ValueError(f"Inconsistent metadata across predictions: {meta_values}")
    meta_row = meta_values.iloc[0].to_dict()

    scores = _pivot_predictions(pred_df, 'score', 'score')
    if 'uncertainty' in pred_df.columns:
        uncertainty = _pivot_predictions(pred_df, 'uncertainty', 'uncertainty')
    else:
        uncertainty = pd.DataFrame(columns=['symbol', 'period', 'time_idx'])

    merged_preds = scores.merge(uncertainty, on=['symbol', 'period', 'time_idx'], how='left')
    for key, value in meta_row.items():
        merged_preds[key] = value

    raw_df = pd.read_pickle(data_path)
    required_cols = {'symbol', 'period', 'time_idx'}
    if not required_cols.issubset(raw_df.columns):
        raise ValueError(f"dataframe from {data_path} lacks required columns {required_cols}")

    targets = target_columns or [c for c in raw_df.columns if c.startswith('target_')]
    label_df = raw_df[['symbol', 'period', 'time_idx'] + targets].drop_duplicates()

    if regime_enabled:
        regime_df = compute_regime_core_features(raw_df)
        label_df = label_df.merge(regime_df, on=['symbol', 'period', 'time_idx'], how='left')

    full_df = merged_preds.merge(label_df, on=['symbol', 'period', 'time_idx'], how='inner')
    full_df.sort_values(['symbol', 'period', 'time_idx'], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for key, value in meta_row.items():
        full_df[key] = value
    full_df.to_parquet(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build OOF dataset for Z-combiner training')
    parser.add_argument('--predictions-root', type=str, default='lightning_logs', help='Root directory containing prediction parquet files')
    parser.add_argument('--data-path', type=str, default='data/pkl_merged/full_merged.pkl', help='Path to merged data (pickle/parquet) with targets')
    parser.add_argument('--output', type=str, default='datasets/z_train.parquet', help='Output parquet path for Z-combiner training set')
    parser.add_argument('--targets', type=str, nargs='*', default=None, help='Explicit target columns to keep (default: all target_*)')
    parser.add_argument('--no-regime', action='store_true', help='Disable regime feature computation')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_root = Path(args.predictions_root)
    data_path = Path(args.data_path)
    output_path = Path(args.output)
    build_oof_dataset(
        predictions_root=predictions_root,
        data_path=data_path,
        output_path=output_path,
        target_columns=args.targets,
        regime_enabled=not args.no_regime,
    )
    print(f"[oof] saved dataset to {output_path}")


if __name__ == '__main__':
    main()
