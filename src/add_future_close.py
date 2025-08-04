def add_future_close_to_dir(input_root, timeframes=["1h", "4h", "1d"], inplace=True):
    import os
    import pandas as pd

    for tf in timeframes:
        dir_path = os.path.join(input_root, tf)
        for fname in os.listdir(dir_path):
            if not fname.endswith(".csv"):
                continue

            fpath = os.path.join(dir_path, fname)
            df = pd.read_csv(fpath)

            if "future_close" in df.columns:
                continue

            if "close" not in df.columns:
                print(f"[跳过] {fname} 缺少 close 列")
                continue

            df["future_close"] = df["close"].shift(-1)
            if inplace:
                df.to_csv(fpath, index=False)
            else:
                df_out = fpath.replace(".csv", "_with_future.csv")
                df.to_csv(df_out, index=False)

            print(f"[✅] 添加 future_close: {fname}")
