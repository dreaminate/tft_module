#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import numpy as np

def check_dataset(path: str, head_n: int = 5):
    # 1. 文件存在与大小
    if not os.path.isfile(path):
        print(f"[ERROR] 文件不存在: {path}")
        sys.exit(1)
    size = os.path.getsize(path)
    if size == 0:
        print(f"[ERROR] 文件大小为 0 字节: {path}")
        sys.exit(1)
    print(f"[INFO] 读取文件: {path} ({size/1024:.1f} KB)")

    # 2. 读取数据
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        print("[ERROR] 仅支持 .parquet 或 .csv 文件")
        sys.exit(1)

    # 3. 空表检查
    if df.empty:
        print(f"[WARNING] 数据表为空 (shape={df.shape})")
        sys.exit(0)

    # 4. 打印基本信息
    print("\n=== 基本信息 ===")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # 5. 尝试将列转为数值，记录失败列
    num_cols = []
    fail_cols = []
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors='raise')
            num_cols.append(col)
        except Exception:
            fail_cols.append(col)
    print(f"\n可转换为数值的列: {len(num_cols)} / {len(df.columns)}")
    if fail_cols:
        print("无法转换的列（可能含非数字值或全空）:", fail_cols)

    # 6. 缺失值 & 无穷大值
    print("\n=== 缺失值统计 ===")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\n=== 无穷大值统计 ===")
    inf = np.isinf(df.select_dtypes(include=[np.number]))
    print(df.select_dtypes(include=[np.number]).columns[inf.any()].tolist() or "无无穷大值")

    # 7. 重复行
    print(f"\n=== 重复行数 === {df.duplicated().sum()}")

    # 8. 数据类型
    print("\n=== 数据类型 ===")
    print(df.dtypes)

    # 9. symbol/period 唯一值（如有）
    for c in ['symbol', 'period']:
        if c in df.columns:
            print(f"\nUnique {c}s:", df[c].dropna().unique().tolist())

    # 10. datetime 连续性
    if 'datetime' in df.columns:
        print("\n=== datetime 连续性（分组） ===")
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        for name, grp in df.groupby([c for c in ['symbol','period'] if c in df.columns]):
            diffs = grp.sort_values('datetime')['datetime'].diff().dt.total_seconds().dropna()
            if not diffs.empty:
                print(f"{name}: min={diffs.min()}s, max={diffs.max()}s, mean={diffs.mean():.1f}s")

    # 11. 描述性统计 & 前 N 行预览
    print("\n=== 描述性统计 (数值列) ===")
    print(df.describe().T)
    print(f"\n=== 前 {head_n} 行 ===")
    print(df.head(head_n))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_dataset_v2.py path/to/data.parquet")
        sys.exit(1)
    check_dataset(sys.argv[1])
