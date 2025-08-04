import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 基础配置 ===
base_dir = "data/crypto_targeted_and_indicated"
symbols = ["BTC_USDT", "ETH_USDT", "BNB_USDT"]
periods = ["1h", "4h", "1d"]

# 最终结果保存路径
output_path = "data/merged/full_merged.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 所有数据列表
all_dfs = []

# 按 period 再按 symbol 顺序加载
for period in periods:
    period_dir = os.path.join(base_dir, period)
    for symbol in symbols:
        file_name = f"{symbol}_{period}_all.csv"
        file_path = os.path.join(period_dir, file_name)
        if not os.path.exists(file_path):
            print(f"[❌] 文件不存在: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df["symbol"] = symbol
        df["period"] = period

        # 删除三分类目标字段
        target_columns_to_remove = [col for col in df.columns if "target_trend3class" in col]
        df.drop(columns=target_columns_to_remove, inplace=True)

        # 确保删除的列确实不存在
        if any("target_trend3class" in col for col in df.columns):
            print(f"[❌] 仍然存在三分类字段: {target_columns_to_remove}")
        else:
            print(f"[✅] 成功删除三分类字段: {target_columns_to_remove}")

        all_dfs.append(df)

# 拼接所有数据
if not all_dfs:
    raise ValueError("❌ 未找到任何数据文件，无法拼接")
merged_df = pd.concat(all_dfs, ignore_index=True)

# 保证每个 group 是连续块，并按时间升序
symbol_order = {sym: i for i, sym in enumerate(symbols)}
period_order = {"1h": 0, "4h": 1, "1d": 2}
merged_df["symbol_rank"] = merged_df["symbol"].map(symbol_order)
merged_df["period_rank"] = merged_df["period"].map(period_order)

# ✅ 正确排序顺序（避免打散 group）
merged_df = merged_df.sort_values(["symbol_rank", "period_rank", "timestamp"]).reset_index(drop=True)
merged_df.drop(columns=["symbol_rank", "period_rank"], inplace=True)

# ✅ 添加 time_idx（按 symbol+period group 累加）
merged_df["time_idx"] = merged_df.groupby(["symbol", "period"]).cumcount()

# 添加检查函数
print("\n[🔎] Group Summary:")
print(merged_df.groupby(["symbol", "period"]).agg({"timestamp": ["min", "max", "count"]}))

def check_group_timeidx(df):
    issues = []
    for (sym, per), group in df.groupby(["symbol", "period"]):
        sorted_ts = group.sort_values("timestamp").reset_index(drop=True)
        expected_idx = list(range(len(sorted_ts)))
        actual_idx = sorted_ts.index.tolist()
        if expected_idx != actual_idx:
            issues.append((sym, per))
    if issues:
        print("[❌] 以下组的 time_idx 可能不连续:", issues)
    else:
        print("[✅] 所有 group 排序正常，time_idx 可安全赋值")

check_group_timeidx(merged_df)

# ✅ 新增绘图：可视化每组样本数量
print("\n[📊] 绘图: 每组样本数量...")
group_counts = merged_df.groupby(["symbol", "period"]).size().reset_index(name="count")
plt.figure(figsize=(8, 4))
sns.barplot(data=group_counts, x="symbol", y="count", hue="period")
plt.title("Sample Counts per (symbol, period)")
plt.ylabel("样本数量")
plt.xlabel("币种")
plt.tight_layout()
plt.savefig("data/merged/group_sample_counts.png")
print("[📈] 样本数量图已保存为 group_sample_counts.png")

# ✅ 新增绘图：每组时间范围 lineplot（datetime）
print("\n[📊] 绘图: 每组样本时间跨度...")
range_df = merged_df.groupby(["symbol", "period"]).agg(
    start=("datetime", "min"), end=("datetime", "max"), count=("datetime", "count")
).reset_index()
plt.figure(figsize=(10, 5))
for _, row in range_df.iterrows():
    plt.plot([row["start"], row["end"]], [f"{row['symbol']}-{row['period']}"] * 2, marker="o")
plt.title("时间跨度 per group")
plt.xlabel("时间")
plt.ylabel("Group")
plt.tight_layout()
plt.savefig("data/merged/group_time_ranges.png")
print("[🕒] 时间跨度图已保存为 group_time_ranges.png")

# ✅ 新增绘图：Span 条带图（datetime） + 样本数量标注
print("\n[📊] 绘图: 每组时间跨度（Span 条带图）...")
plt.figure(figsize=(10, 6))
for i, row in range_df.iterrows():
    plt.hlines(y=i, xmin=row["start"], xmax=row["end"], linewidth=8)
    plt.text(row["start"], i + 0.2, f"{row['symbol']}-{row['period']} ({row['count']})", fontsize=8)
plt.yticks([])
plt.xlabel("时间")
plt.title("时间覆盖范围 per group")
plt.tight_layout()
plt.savefig("data/merged/group_time_spans.png")
print("[📏] Span 条带图已保存为 group_time_spans.png")

# 保存结果
merged_df.to_csv(output_path, index=False)
print(f"[✅] 融合完成，保存至: {output_path}")
#  python src/merged.py