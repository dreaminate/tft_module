# generate_targets_auto.py
import os
import pandas as pd
import numpy as np
from target_config import process_period_targets
from add_future_close import add_future_close_to_dir

# === 配置项 ===
INPUT_ROOT = "data/crypto_indicated"                      # 输入：技术指标数据
OUTPUT_ROOT = "data/crypto_targeted_and_indicated"        # 输出：含目标字段
FUTURE_COL = "future_close"                               # 用于计算 logreturn 的未来价格列
PERIODS = ["1h", "4h", "1d"]                              # 周期列表

# ---------- 小工具 ----------
def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保存在 'timestamp' 列：
    - 若已有，直接返回
    - 若无但有 'datetime'（datetime64），则生成毫秒级 timestamp
    - 两者都没有就抛错
    """
    if "timestamp" in df.columns:
        return df
    if "datetime" in df.columns:
        if not np.issubdtype(df["datetime"].dtype, np.datetime64):
            # 尝试解析成时间
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        if df["datetime"].isna().all():
            raise ValueError("无法从 'datetime' 解析时间，且缺少 'timestamp'。")
        df["timestamp"] = (df["datetime"].view("int64") // 10**6).astype("int64")  # ms
        return df
    raise KeyError("输入缺少 'timestamp' 和 'datetime'，无法继续。")

def cast_numeric32(df: pd.DataFrame) -> pd.DataFrame:
    """把 float64→float32, int64→int32（不动类别/时间列），省内存更快。"""
    floats = df.select_dtypes(include=["float64"]).columns
    ints   = [c for c in df.select_dtypes(include=["int64"]).columns if c not in ("timestamp",)]
    df[floats] = df[floats].astype("float32")
    df[ints]   = df[ints].astype("int32")
    return df

# ---------- 主流程 ----------
def convert_selected_periods_to_csv(src_root: str, dst_root: str, selected_periods=None):
    if selected_periods is None:
        selected_periods = PERIODS

    # 先给所有周期补齐 future_close（若不存在）
    add_future_close_to_dir(src_root, timeframes=selected_periods, inplace=True)

    # 处理每个周期
    for period in selected_periods:
        input_dir = os.path.join(src_root, period)
        output_dir = os.path.join(dst_root, period)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isdir(input_dir):
            print(f"[跳过] 目录不存在：{input_dir}")
            continue

        for fname in os.listdir(input_dir):
            if not fname.endswith(".csv"):
                continue
            file_path = os.path.join(input_dir, fname)

            try:
                # 强制解析 datetime，保证后续能构造 timestamp
                df = pd.read_csv(file_path, parse_dates=["datetime"], low_memory=False)

                # 统一字段名（有些 csv 可能带 symbol 后缀）
                symbol_name = fname.replace(".csv", "").replace(f"_{period}_all", "")
                for col in ["open", "high", "low", "close", "volume"]:
                    suffixed = f"{col}_{symbol_name}_{period}"
                    if suffixed in df.columns:
                        df.rename(columns={suffixed: col}, inplace=True)

                # 必要列检查
                need_cols = {"open","high","low","close","volume"}
                missing = [c for c in need_cols if c not in df.columns]
                if missing:
                    print(f"[跳过] {fname} 缺少必需列: {missing}")
                    continue

                # 确保 timestamp 列存在
                df = ensure_timestamp(df)

                # 确保 period/symbol 列存在（有些管线可能没存 period）
                if "period" not in df.columns:
                    df["period"] = period
                if "symbol" not in df.columns:
                    df["symbol"] = symbol_name

                # 数值类型收敛到 float32/int32
                df = cast_numeric32(df)

                # 检查 future_close
                if FUTURE_COL not in df.columns:
                    print(f"[跳过] {fname} 仍缺少列 {FUTURE_COL}（检查 add_future_close_to_dir）")
                    continue

                print(f"[🚀] 处理: {fname} | 周期: {period} | 币种: {symbol_name} | 行数: {len(df)}")

                # 目标构造
                df_out = process_period_targets(
                    df.copy(), period=period, future_col=FUTURE_COL, symbol_name=symbol_name
                )

                # 保存输出（保持原始文件名）
                out_path = os.path.join(output_dir, fname)
                df_out.to_csv(out_path, index=False)
                print(f"[✅] 保存: {out_path} | 行数: {len(df_out)}")

            except Exception as e:
                print(f"[❌] 处理失败: {fname} | 原因: {e}")

    print("✅ 所有选定周期目标构造完成")

# === 执行入口 ===
if __name__ == "__main__":
    selected_periods = ["1d", "1h", "4h"]
    convert_selected_periods_to_csv(INPUT_ROOT, OUTPUT_ROOT, selected_periods)
    print(f"✅ 全部完成，输出目录: {OUTPUT_ROOT}")
