"""改周期之类的，去文件里改"""
from apied import apied_main
from ccatch import catch_main
from generate_targets_auto import convert_selected_periods_to_csv
from indicating import indicating_main
from merged import merged_main
from csv2Parquet import csv2Parquet_main
from csv2Pkl import csv2Pkl_main
periods = ["1h", "4h", "1d"]
# ===链上数据抓取===
apied_main()
# ===代币数据抓取===
catch_main()
# ===技术指标计算===
for period in periods:
    indicating_main(period)
# ===目标构造===
convert_selected_periods_to_csv("data/crypto_indicated", "data/crypto_targeted_and_indicated", selected_periods=periods)

#   python src/all_process_data.py