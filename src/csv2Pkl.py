# src/csv2Pkl.py

import pandas as pd
from pathlib import Path

# === 配置路径 ===
# SRC_FILE = Path("data/merged/full_merged.pruned.csv")      # 要转换的 CSV 文件
SRC_FILE = Path("data/merged/full_merged.csv")      # 要转换的 CSV 文件
DST_FILE = Path("data/pkl_merged/full_merged.pkl")  # 输出 pkl 文件路径
SELECTED_PERIODS = ["1h", "4h", "1d"]  # 你想保留的周期，None 表示不过滤

def convert_full_merged_csv_to_pkl(src_file: Path, dst_file: Path, selected_periods=None):
    if not src_file.exists():
        print(f"[❌] 源文件不存在: {src_file}")
        return

    try:
        print(f"\n▶ 正在处理: {src_file}")
        df = pd.read_csv(src_file)

        # === 检查必须字段 ===
        if "timestamp" not in df.columns:
            print("❌ 缺失 timestamp 字段")
            return

        # 构造 datetime
        df["datetime"] = pd.to_datetime(
            df["timestamp"],
            unit="ms" if df["timestamp"].max() > 1e12 else None,
            errors="coerce"
        )

        # 周期过滤
        if selected_periods and "period" in df.columns:
            before = len(df)
            df = df[df["period"].isin(selected_periods)]
            print(f"🔍 周期过滤: {before} → {len(df)}")

        if df.empty:
            print("⚠️ 文件数据为空（过滤后）")
        else:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(dst_file)

            print(f"✅ 转换成功 → {dst_file.name}")
            print(f"   📊 总行数: {len(df)}")
            print(f"   🧪 周期分布: {df['period'].value_counts().to_dict() if 'period' in df.columns else '无'}")
            print(f"   🔍 含 NaN 比例: {(df.isna().any(axis=1).mean() * 100):.2f}%")
            nan_cols = df.columns[df.isnull().any()]
            print(df[nan_cols].isnull().mean().sort_values(ascending=False))

    except Exception as e:
        print(f"❌ 转换失败: {e}")

if __name__ == "__main__":
    convert_full_merged_csv_to_pkl(SRC_FILE, DST_FILE, SELECTED_PERIODS)
#  python src/csv2Pkl.py