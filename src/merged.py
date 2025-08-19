import os
import warnings
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================
# 工具函数
# ============================================

def load_csv(file_path: str) -> pd.DataFrame:
    """读取单个 symbol 周期 csv，并统一处理时间列 & 无效列。"""
    # 先快速探查列名
    cols = pd.read_csv(file_path, nrows=0).columns
    parse_cols = ["datetime"] if "datetime" in cols else []
    df = pd.read_csv(file_path, parse_dates=parse_cols)

    # 若仅有 timestamp(ms)，补建 datetime
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # 清理 Unnamed 空列
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True)
    return df


def ensure_time_idx(df: pd.DataFrame) -> None:
    """检查 time_idx 连续性，打印异常 group。"""
    issues = []
    for (sym, per), g in df.groupby(["symbol", "period"]):
        if not (g["time_idx"].diff().fillna(1) == 1).all():
            issues.append((sym, per))
    if issues:
        print("❌ time_idx 不连续:", issues)
    else:
        print("✅ time_idx 全部连续")


# ============================================
# 主逻辑
# ============================================

def main(base_dir: str, output_path: str, symbols: list[str], periods: list[str]):
    all_dfs = []
    for period in periods:
        period_dir = os.path.join(base_dir, period)
        for symbol in symbols:
            file_name = f"{symbol}_{period}.csv"
            file_path = os.path.join(period_dir, file_name)
            if not os.path.exists(file_path):
                print(f"[❌] 文件不存在: {file_path}")
                continue

            df = load_csv(file_path)
            df["symbol"] = symbol
            df["period"] = period

            # 删除三分类目标列
            tri_cols = [c for c in df.columns if "target_trend3class" in c]
            df.drop(columns=tri_cols, inplace=True)
            if tri_cols:
                print(f"[✅] 删除三分类字段: {tri_cols}")

            all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError("未找到任何有效 csv，终止拼接")

    merged = pd.concat(all_dfs, ignore_index=True)

    # 排序确保 group 连贯
    sym_order = {s: i for i, s in enumerate(symbols)}
    per_order = {p: i for i, p in enumerate(periods)}
    merged["symbol_rank"] = merged["symbol"].map(sym_order)
    merged["period_rank"] = merged["period"].map(per_order)
    merged.sort_values(["symbol_rank", "period_rank", "timestamp"], inplace=True)
    merged.drop(columns=["symbol_rank", "period_rank"], inplace=True)

    # time_idx
    merged["time_idx"] = merged.groupby(["symbol", "period"]).cumcount()

    print("\n[🔎] Group Summary:")
    print(
        merged.groupby(["symbol", "period"]).agg(
            start_ts=("datetime", "min"), end_ts=("datetime", "max"), samples=("datetime", "count")
        )
    )

    ensure_time_idx(merged)

    # --- 绘图设置（防中文乱码） ---
    plt.rcParams["font.family"] = ["Arial", "sans-serif"]

    # 样本数量柱状图
    counts_df = merged.groupby(["symbol", "period"]).size().reset_index(name="count")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=counts_df, x="symbol", y="count", hue="period", order=symbols, hue_order=periods)
    plt.title("Sample Counts per (symbol, period)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts_path = os.path.join(os.path.dirname(output_path), "group_sample_counts.png")
    plt.savefig(counts_path)
    plt.close()
    print(f"[📈] 样本数量图保存: {counts_path}")

    # 时间跨度线段图
    span_df = merged.groupby(["symbol", "period"]).agg(
        start=("datetime", "min"), end=("datetime", "max"), cnt=("datetime", "count")
    ).reset_index()
    plt.figure(figsize=(10, 5))
    for _, row in span_df.iterrows():
        plt.plot([row["start"], row["end"]], [f"{row['symbol']}-{row['period']}"] * 2, marker="o")
    plt.title("时间跨度 per group")
    plt.xlabel("时间")
    plt.ylabel("Group")
    plt.tight_layout()
    range_path = os.path.join(os.path.dirname(output_path), "group_time_ranges.png")
    plt.savefig(range_path)
    plt.close()
    print(f"[🕒] 时间跨度图保存: {range_path}")

    # Span 条带图
    plt.figure(figsize=(10, 6))
    for i, row in span_df.iterrows():
        plt.hlines(y=i, xmin=row["start"], xmax=row["end"], linewidth=8)
        plt.text(row["start"], i + 0.2, f"{row['symbol']}-{row['period']} ({row['cnt']})", fontsize=8)
    plt.yticks([])
    plt.xlabel("时间")
    plt.title("时间覆盖范围 per group")
    plt.tight_layout()
    span_path = os.path.join(os.path.dirname(output_path), "group_time_spans.png")
    plt.savefig(span_path)
    plt.close()
    print(f"[📏] Span 图保存: {span_path}")

    # 保存合并结果
    merged.to_csv(output_path, index=False)
    print(f"[✅] 数据融合完成 → {output_path}\n")


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge crypto CSVs across symbols & periods")
    parser.add_argument("--base_dir", default="data/crypto_targeted_and_indicated_merged", help="根目录，里面按 period/SYMBOL.csv 存放")
    parser.add_argument("--output", default="data/merged/full_merged.csv", help="输出 merged csv 路径")
    parser.add_argument("--symbols", nargs="*", default=["BTC_USDT", "ETH_USDT", "BNB_USDT","ADA_USDT","SOL_USDT"], help="币种列表")
    parser.add_argument("--periods", nargs="*", default=["1h", "4h", "1d"], help="周期列表")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    main(args.base_dir, args.output, args.symbols, args.periods)
# python src/merged.py --base_dir data/crypto_targeted_and_indicated_merged --output data/merged/full_merged.csv --symbols BTC_USDT ETH_USDT BNB_USDT ADA_USDT SOL_USDT --periods 1h 4h 1d