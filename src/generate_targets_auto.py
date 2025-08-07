import os
import pandas as pd
from target_config import process_period_targets               # ✅ 替换为封装函数
from add_future_close import add_future_close_to_dir          # ✅ 引入 future close 补全


# === 配置项 ===
INPUT_ROOT = "data/crypto_indicated"                            # 输入：技术指标数据
OUTPUT_ROOT = "data/crypto_targeted_and_indicated"              # 输出：含目标字段
FUTURE_COL = "future_close"                                     # 用于计算 logreturn 的未来价格列
PERIODS = ["1h", "4h", "1d"]                                     # 周期列表

# ✅ 第一步：先为所有周期添加 future_close 字段（如不存在则补齐）
add_future_close_to_dir(INPUT_ROOT, timeframes=PERIODS, inplace=True)

# ✅ 第二步：执行目标构造 + 清洗 + 输出
def convert_selected_periods_to_parquet(src_root: str, dst_root: str, selected_periods=None):
    """
    仅转换选定周期的数据。
    selected_periods: 可选，若为 None 则转换所有周期，否则仅转换指定周期的数据。
    """
    if selected_periods is None:
        selected_periods = PERIODS  # 如果没有指定周期，则转换所有周期

    # 处理每个周期的数据
    for period in selected_periods:
        input_dir = os.path.join(src_root, period)
        output_dir = os.path.join(dst_root, period)
        os.makedirs(output_dir, exist_ok=True)

        for fname in os.listdir(input_dir):
            if not fname.endswith(".csv"):
                continue

            file_path = os.path.join(input_dir, fname)
            df = pd.read_csv(file_path)

            if FUTURE_COL not in df.columns:
                print(f"[跳过] {fname} 缺少列 {FUTURE_COL}")
                continue

            symbol_name = fname.replace(".csv", "").replace(f"_{period}_all", "")
            print(f"[🚀] 开始处理: {fname} | 周期: {period} | 币种: {symbol_name}")

            # 调用封装函数，包含构造 + 清洗 + 打印
            df_targets = process_period_targets(df.copy(), period, future_col=FUTURE_COL, symbol_name=symbol_name)

            # ✅ 保存结果
            
            out_path = os.path.join(output_dir, fname)
           
            df_targets.to_csv(out_path, index=False)

    print("✅ 所有周期目标构造完成")

# === 执行入口 ===
if __name__ == "__main__":
    selected_periods = ["1d","1h","4h"]  # 只转换 1d 周期的数据（可以根据需要修改为多个周期）
    convert_selected_periods_to_parquet(INPUT_ROOT, OUTPUT_ROOT, selected_periods)
    print(f"✅ 所有选定周期目标构造完成，输出目录: {OUTPUT_ROOT}")
#         python src/generate_targets_auto.py