import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


DEFAULT_TARGETS_CLS = ["target_binarytrend"]
DEFAULT_TARGETS_REG = ["target_logreturn"]


def list_candidate_features(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = {"timestamp", "time_idx"}
    drop_prefix = ("future",)
    cand = []
    for c in num_cols:
        if c in drop_cols:
            continue
        if any(c.startswith(p) for p in drop_prefix):
            continue
        if c.startswith("target_"):
            continue
        cand.append(c)
    # 也允许少量重要的非数值列在外部加入，但默认仅数值
    return cand


def fast_nan_filter(df: pd.DataFrame, cols: list[str], min_coverage: float) -> list[str]:
    keep = []
    thresh = 1.0 - float(min_coverage)
    for c in cols:
        na_ratio = float(df[c].isna().mean())
        if na_ratio <= thresh:
            keep.append(c)
    return keep


def drop_mm_keep_zn(cols: list[str]) -> list[str]:
    # 丢弃 *_mm*，保留 *_zn*
    return [c for c in cols if "_mm" not in c]


def drop_high_corr(df: pd.DataFrame, cols: list[str], sample_n: int = 50000, corr_th: float = 0.98, seed: int = 42) -> list[str]:
    if len(cols) <= 1:
        return cols
    sub = df[cols].dropna(how="any")
    if len(sub) == 0:
        return cols
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=seed)
    # 计算 Spearman 更稳健（如需改可改为 pearson）
    corr = sub.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for c in cols:
        if c in to_drop:
            continue
        high = upper[c][upper[c] > corr_th].index.tolist() if c in upper else []
        for h in high:
            to_drop.add(h)
    pruned = [c for c in cols if c not in to_drop]
    return pruned


def rank_features_mi(df: pd.DataFrame, X_cols: list[str], target: str, task: str, sample_n: int = 80000, seed: int = 42) -> pd.DataFrame:
    # 仅用训练期样本，避免泄露：这里简单用最早 80% 的时间
    df = df.sort_values(["symbol", "period", "datetime"]) if "datetime" in df.columns else df.copy()
    # 简单切分：按全局行号 0.8 分
    cut = int(len(df) * 0.8)
    df_tr = df.iloc[:cut]

    sub = df_tr[X_cols + [target]].dropna(how="any")
    if len(sub) == 0:
        return pd.DataFrame(columns=["feature", "mi"])  # 空
    if len(sub) > sample_n:
        sub = sub.sample(sample_n, random_state=seed)

    X = sub[X_cols].values
    y = sub[target].values
    # 分类/回归
    if task == "cls":
        y = (y > 0.5).astype(int) if y.dtype.kind != "i" else y.astype(int)
        scores = mutual_info_classif(X, y, random_state=seed, discrete_features=False)
    else:
        scores = mutual_info_regression(X, y, random_state=seed)
    out = pd.DataFrame({"feature": X_cols, "mi": scores}).sort_values("mi", ascending=False).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pkl_merged/full_merged.pkl")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--min_coverage", type=float, default=0.85, help="最小覆盖率（非缺失占比）")
    ap.add_argument("--drop_mm", action="store_true", help="丢弃 *_mm* 特征，仅保留 *_zn* 和 raw")
    ap.add_argument("--out", default="configs/selected_features.txt")
    ap.add_argument("--targets_cls", default=",".join(DEFAULT_TARGETS_CLS))
    ap.add_argument("--targets_reg", default=",".join(DEFAULT_TARGETS_REG))
    ap.add_argument("--sample_corr", type=int, default=50000)
    ap.add_argument("--corr_th", type=float, default=0.98)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.read_pickle(args.data)

    # 候选特征集合
    cand = list_candidate_features(df)
    # 覆盖率筛选
    cand = fast_nan_filter(df, cand, min_coverage=args.min_coverage)
    # 可选：丢弃 mm 版本
    if args.drop_mm:
        cand = drop_mm_keep_zn(cand)
    # 高相关去冗余
    cand = drop_high_corr(df, cand, sample_n=args.sample_corr, corr_th=args.corr_th)

    # 目标集合
    t_cls = [t for t in args.targets_cls.split(",") if t]
    t_reg = [t for t in args.targets_reg.split(",") if t]

    keep_union: set[str] = set()

    for t in t_cls:
        if t not in df.columns:
            continue
        rank = rank_features_mi(df, cand, target=t, task="cls")
        keep = rank.head(args.topk)["feature"].tolist()
        keep_union.update(keep)

    for t in t_reg:
        if t not in df.columns:
            continue
        rank = rank_features_mi(df, cand, target=t, task="reg")
        keep = rank.head(args.topk)["feature"].tolist()
        keep_union.update(keep)

    selected = sorted(list(keep_union))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# auto-generated feature allowlist\n")
        for f in selected:
            fh.write(f"{f}\n")

    print(f"[FS] selected features: {len(selected)} -> {args.out}")


if __name__ == "__main__":
    main()

