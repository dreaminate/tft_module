from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def audit_no_leakage(path: Path, max_gap: int = 1000) -> bool:
    df = pd.read_parquet(path)
    required_cols = {'symbol', 'period', 'time_idx'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    status = []
    ok = True

    duplicates = df.duplicated(['symbol', 'period', 'time_idx']).sum()
    status.append(("duplicate_rows", int(duplicates)))
    if duplicates:
        ok = False

    negative_steps = 0
    gap_list = []
    for (_, _), group in df.groupby(['symbol', 'period'], sort=False):
        ordered = group.sort_values('time_idx')
        diff = ordered['time_idx'].diff().dropna()
        negative_steps += int((diff <= 0).sum())
        if not diff.empty:
            gap_list.append(float(diff.max()))
    status.append(("non_positive_steps", int(negative_steps)))
    if negative_steps:
        ok = False
    if gap_list:
        max_gap_actual = max(gap_list)
        status.append(("max_gap", max_gap_actual))
        if max_gap_actual > max_gap:
            ok = False
    else:
        status.append(("max_gap", float('nan')))

    if 'train_window_id' in df.columns:
        status.append(("train_window_ids", df['train_window_id'].value_counts().to_dict()))

    if {'schema_ver', 'data_ver', 'expert_ver'} <= set(df.columns):
        versions = df[['schema_ver', 'data_ver', 'expert_ver']].drop_duplicates().to_dict('records')
        status.append(("versions", versions))

    score_cols = [c for c in df.columns if c.startswith('score__')]
    if score_cols:
        missing_scores = df[score_cols].isna().sum().sum()
        status.append(("missing_scores", int(missing_scores)))
        if missing_scores:
            ok = False

    print("=== Leakage Audit Summary ===")
    for name, value in status:
        print(f"{name}: {value}")
    print("PASS" if ok else "FAIL")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audit z_train.parquet for leakage risks')
    parser.add_argument('--path', type=str, default='datasets/z_train.parquet')
    parser.add_argument('--max-gap', type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok = audit_no_leakage(Path(args.path), max_gap=args.max_gap)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
