import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
MERGED_PATH = os.path.join(BASE_DIR, '..', 'data', 'merged', 'full_merged.csv')

def check_merged_data(path):
    if not os.path.exists(path):
        print(f"❌ 找不到文件：{path}")
        return

    df = pd.read_csv(path)
    print(f"✅ 成功读取文件，共 {len(df)} 条记录")

    # ✅ 1. 基本列检查
    expected_cols = [
        'timestamp', 'symbol', 'close', 'future_close_4h',
        'target_return', 'target_updown',
        'target_volatility', 'target_risk_adj_return'
    ]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        print(f"❌ 缺少以下列：{missing}")
        return
    else:
        print("✅ 所有关键列都存在")

    # ✅ 2. symbol 分布
    print("📊 各 symbol 分布：")
    print(df['symbol'].value_counts())

    # ✅ 3. 时间是否递增
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    is_sorted = df.sort_values(['timestamp', 'symbol'])['timestamp'].is_monotonic_increasing
    print(f"🕒 时间是否递增排序：{'✅ 是' if is_sorted else '❌ 否'}")

    # ✅ 4. target_updown 检查
    updown_vals = set(df['target_updown'].unique())
    print(f"🔺 target_updown 唯一值: {updown_vals}")
    if not updown_vals.issubset({0, 1}):
        print("⚠️ 存在异常值！")

    # ✅ 5. target_return / risk 分布
    print(f"📈 target_return 描述：\n{df['target_return'].describe()}")
    print(f"📉 target_risk_adj_return 描述：\n{df['target_risk_adj_return'].describe()}")

    # ✅ 6. 缺失值检查
    null_summary = df[expected_cols].isnull().sum()
    print("🕳️ 缺失值情况：")
    print(null_summary[null_summary > 0])

if __name__ == "__main__":
    check_merged_data(MERGED_PATH)
