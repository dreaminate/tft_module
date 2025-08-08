# feature_healthcheck.py
import os
import pandas as pd
import numpy as np

def feature_report(
    df: pd.DataFrame,
    group_cols=("symbol","period"),
    zn_tag="_zn",              # 你的归一化后缀关键词（会自动匹配 _zn48 / _zn30 等）
    mm_tag="_mm",
    top_corr=30,               # 最多展示的高相关对
    save_dir=None,             # 如果给路径，就保存 csv
):
    rep = {}

    # -------- 1) 基础信息 --------
    rep["n_rows"] = len(df)
    rep["n_groups"] = df.groupby(list(group_cols), sort=False).ngroups

    # 只看数值列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 标记 raw / zn / mm
    zn_cols = [c for c in num_cols if zn_tag in c]
    mm_cols = [c for c in num_cols if mm_tag in c]
    # 常见 raw 特征，按你的管线挑一些关键的
    RAW_PREFIX = ("open","high","low","close","volume","ma","rsi","macd","kdj","obv",
                  "boll","atr","trendline","volume_relative","distance_to_",
                  "support_level","resistance_level","vwap","cci","adx")
    raw_cols = [c for c in num_cols
                if (c.startswith(RAW_PREFIX) and (zn_tag not in c) and (mm_tag not in c))]

    # -------- 2) 缺失 / 常数 / 极端值 --------
    def _basic_stats(cols):
        sub = df[cols].copy()
        na_ratio = sub.isna().mean().sort_values(ascending=False)
        nunique = sub.nunique().sort_values()
        const_cols = nunique[nunique <= 1].index.tolist()

        # 极端值：|z|>6 的占比粗查（对每列做 group-wise 粗 z，不严格）
        # 这里用全局 z 近似即可，避免太慢
        z = (sub - sub.mean()) / (sub.std().replace(0, np.nan))
        extreme_ratio = (z.abs() > 6).mean().sort_values(ascending=False)

        return na_ratio, nunique, const_cols, extreme_ratio

    na_raw, nu_raw, const_raw, ex_raw = _basic_stats(raw_cols)
    na_zn,  nu_zn,  const_zn,  ex_zn  = _basic_stats(zn_cols)
    na_mm,  nu_mm,  const_mm,  ex_mm  = _basic_stats(mm_cols)

    rep["raw_na_top"] = na_raw.head(20)
    rep["zn_na_top"]  = na_zn.head(20)
    rep["mm_na_top"]  = na_mm.head(20)

    rep["raw_const"] = const_raw
    rep["zn_const"]  = const_zn
    rep["mm_const"]  = const_mm

    rep["raw_extreme_top"] = ex_raw.head(20)
    rep["zn_extreme_top"]  = ex_zn.head(20)
    rep["mm_extreme_top"]  = ex_mm.head(20)

    # -------- 3) 分布对比：raw vs zn 同源列（可快速 eyeballing）--------
    # 例如 close 与 close_znXX 的均值/方差/分位
    def _summary(series):
        q = series.quantile([0.01,0.05,0.25,0.5,0.75,0.95,0.99])
        return pd.Series({
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p1": float(q.loc[0.01]),
            "p5": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]),
            "p50": float(q.loc[0.5]),
            "p75": float(q.loc[0.75]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
            "max": float(series.max()),
        })

    pairs = []
    for rc in raw_cols:
        # 找匹配的 zn 列（可能有 _zn48/_zn30 多个，就选一个长度最长的窗口名也行）
        cand = [c for c in zn_cols if c.startswith(rc + "_") or c.startswith(rc) and zn_tag in c]
        if not cand:
            continue
        # 选第一个匹配（你的命名统一的话足够）
        zc = cand[0]
        pairs.append((rc, zc))
    dist_rows = []
    for (rc, zc) in pairs[:100]:  # 限个数，避免巨慢
        s_raw = df[rc].dropna()
        s_zn  = df[zc].dropna()
        if len(s_raw) < 50 or len(s_zn) < 50:
            continue
        sr = _summary(s_raw)
        sz = _summary(s_zn)
        row = pd.concat([sr.add_prefix("raw_"), sz.add_prefix("zn_")])
        row.name = f"{rc} -> {zc}"
        dist_rows.append(row)
    rep["raw_vs_zn_summary"] = pd.DataFrame(dist_rows).sort_index()

    # -------- 4) 高相关冗余检查（在 zn 空间）--------
    highcorr = None
    if len(zn_cols) >= 2:
        # 用样本较全的行，避免 NA 影响
        sub = df[zn_cols].copy()
        sub = sub.dropna(axis=0, how="any")
        if len(sub) > 1000:
            sub = sub.sample(1000, random_state=42)
        corr = sub.corr().abs()
        # 拿上三角
        tri = (np.triu(np.ones(corr.shape), k=1) == 1)
        corr_vals = corr.where(tri)
        pairs_corr = (
            corr_vals.stack()
            .sort_values(ascending=False)
            .head(top_corr)
            .to_frame("abs_corr")
        )
        highcorr = pairs_corr
    rep["zn_highcorr_pairs"] = highcorr

    # -------- 5) 每组早期 NaN 比率（shift+rolling导致的前几行缺失）--------
    early_nan = []
    for gkey, gdf in df.groupby(list(group_cols), sort=False):
        head = gdf.head(64)  # 看前 64 根
        nan_rate = head[zn_cols + mm_cols].isna().mean().mean() if (zn_cols or mm_cols) else np.nan
        early_nan.append({"group": gkey, "early_nan_rate_avg": nan_rate})
    rep["early_nan_by_group"] = pd.DataFrame(early_nan).sort_values("early_nan_rate_avg", ascending=False)

    # -------- 6) 输出/保存 --------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        def _save(name, obj):
            path = os.path.join(save_dir, f"{name}.csv")
            if isinstance(obj, pd.Series):
                obj.to_csv(path, header=True)
            elif isinstance(obj, pd.DataFrame):
                obj.to_csv(path, index=True)
        _save("raw_na_top", rep["raw_na_top"])
        _save("zn_na_top",  rep["zn_na_top"])
        _save("mm_na_top",  rep["mm_na_top"])
        _save("raw_vs_zn_summary", rep["raw_vs_zn_summary"])
        if rep["zn_highcorr_pairs"] is not None:
            _save("zn_highcorr_pairs", rep["zn_highcorr_pairs"])
        _save("early_nan_by_group", rep["early_nan_by_group"])

    return rep


if __name__ == "__main__":
    # 示例：指向你合并后的 pkl
    path = "data/pkl_merged/full_merged.pkl"
    df = pd.read_pickle(path)

    # 只取 1 个或少量 symbol/period 快速跑（第一次建议这样）
    # df = df[(df["symbol"].isin(["BTC_USDT","ETH_USDT"])) & (df["period"]=="1h")]

    rep = feature_report(
        df,
        group_cols=("symbol","period"),
        zn_tag="_zn", mm_tag="_mm",
        top_corr=40,
        save_dir="feature_health_reports"
    )

    # 终端上简要打印几项
    print("=== 常数列(raw/zn/mm) ===")
    print("raw_const:", rep["raw_const"][:10])
    print("zn_const:",  rep["zn_const"][:10])
    print("mm_const:",  rep["mm_const"][:10])
    print("\n=== 缺失率TOP(zn) ===")
    print(rep["zn_na_top"].head(10))
    print("\n=== zn高相关特征对 ===")
    print(rep["zn_highcorr_pairs"].head(10) if rep["zn_highcorr_pairs"] is not None else "N/A")
    print("\n=== 组内前64根平均NaN比率（归一化导致） ===")
    print(rep["early_nan_by_group"].head(10))
    # python src/feature_healthcheck.py