from __future__ import annotations
import argparse
import os
from typing import Dict
import pandas as pd


def suggest_quality_by_missing(df: pd.DataFrame, period_col: str = "period", rich_cols_prefix: str = "onchain_") -> Dict[str, float]:
    if period_col not in df.columns:
        raise ValueError(f"missing column: {period_col}")
    # 简单策略：按 period 统计富模态列的非缺失率均值，线性缩放到 [0.6, 1.0]
    cols = [c for c in df.columns if c.startswith(rich_cols_prefix)]
    if not cols:
        return {}
    g = df.groupby(period_col)[cols].apply(lambda x: 1.0 - x.isna().mean())
    score = g.mean(axis=1)
    # 映射到 [0.6, 1.0]
    mn = float(score.min()) if len(score) else 0.0
    mx = float(score.max()) if len(score) else 1.0
    out: Dict[str, float] = {}
    for per, val in score.items():
        if mx > mn:
            q = 0.6 + 0.4 * (float(val) - mn) / (mx - mn)
        else:
            q = 1.0
        out[str(per)] = float(max(0.0, min(1.0, q)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Suggest rich_quality_weights by missing-ratio heuristics")
    ap.add_argument("--pkl", type=str, required=True, help="Path to merged dataset PKL with period + rich columns")
    ap.add_argument("--prefix", type=str, default="onchain_", help="Prefix of rich modality columns")
    args = ap.parse_args()
    assert os.path.exists(args.pkl), f"file not found: {args.pkl}"
    df: pd.DataFrame = pd.read_pickle(args.pkl)
    weights = suggest_quality_by_missing(df, period_col="period", rich_cols_prefix=args.prefix)
    if not weights:
        print("{}")
        return
    import json
    print(json.dumps(weights, ensure_ascii=False))


if __name__ == "__main__":
    main()


