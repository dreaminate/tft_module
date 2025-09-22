import os
import argparse
from typing import List

from indicating import indicating_main
from generate_targets_auto import convert_selected_periods_to_csv
from merged import merged_main


def _detect_symbols(base_dir: str, periods: List[str]) -> List[str]:
    symbols = set()
    for p in periods:
        pdir = os.path.join(base_dir, p)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if not fn.endswith(".csv"):
                continue
            suf = f"_{p}_all.csv"
            if fn.endswith(suf):
                sym = fn[: -len(suf)]
                if sym:
                    symbols.add(sym)
    return sorted(symbols)


def main():
    ap = argparse.ArgumentParser(description="Build full_merged.csv: indicators -> targets -> merge")
    # 支持：
    #   --periods 1h 4h 1d
    #   --periods "1h,4h,1d"
    #   --periods 1h,4h,1d  （PowerShell 下推荐加引号）
    ap.add_argument("--periods", nargs="+", default=["1h","4h","1d"], help="Periods: space or comma separated (e.g. 1h 4h 1d or '1h,4h,1d')")
    ap.add_argument("--indicated-root", type=str, default=os.path.join("data", "crypto_indicated"))
    ap.add_argument("--targeted-root", type=str, default=os.path.join("data", "crypto_targeted_and_indicated"))
    ap.add_argument("--merged-out", type=str, default=os.path.join("data", "merged", "full_merged.csv"))
    args = ap.parse_args()

    raw_periods = args.periods if isinstance(args.periods, list) else [str(args.periods)]
    parts = []
    for token in raw_periods:
        if token is None:
            continue
        s = str(token)
        # 进一步拆分逗号形式
        for seg in s.split(','):
            seg = seg.strip()
            if seg:
                parts.append(seg)
    # 允许的周期集合（避免误传如 '1'）
    allowed = {"1h","4h","1d"}
    periods = [p for p in parts if p in allowed]
    if not periods:
        periods = ["1h","4h","1d"]

    # 1) 计算指标（基于 data/crypto/<period> 输入）
    for p in periods:
        print(f"[indicators] period={p}")
        indicating_main(p)

    # 2) 生成目标 + 输出到 targeted 目录
    print("[targets] converting indicated -> targeted_and_indicated ...")
    convert_selected_periods_to_csv(args.indicated_root, args.targeted_root, selected_periods=periods)

    # 3) 合并为 full_merged.csv
    print("[merge] building full_merged.csv ...")
    symbols = _detect_symbols(args.targeted_root, periods)
    if not symbols:
        raise SystemExit("No symbols detected under targeted root; aborting")
    os.makedirs(os.path.dirname(args.merged_out), exist_ok=True)
    merged_main(args.targeted_root, args.merged_out, symbols, periods)
    print(f"[done] saved -> {os.path.abspath(args.merged_out)}")


if __name__ == "__main__":
    main()


