from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Optional, Tuple
import re

# ===================== 配置：你给的两批列 =====================

# 1) 先整列删除（“太高”的这些）
DROP_COLS = [
    "x_3_golden_ratio_multiplier",
    "cgdi_index_value_futures_cgdi",
    "interest_rate_borrow_interest_rate_binance",
    "open_fr_vol", "low_fr_vol", "close_fr_vol", "high_fr_vol",
    "cdri_index_value_futures_cdri",
    "tweet_count_bitcoin_bubble_index",
    "address_send_count_bitcoin_bubble_index",
    "mining_difficulty_bitcoin_bubble_index",
    "google_trend_percent_bitcoin_bubble_index",
    "bubble_index_bitcoin_bubble_index",
    "transaction_count_bitcoin_bubble_index",
]

# 2) 其余这些列：按缺失删行（以前默认 any=任一缺失就删，**极易误伤**）。
ROW_FILTER_COLS = [
    "global_m2_supply_bitcoin_bitcoin_vs_global_m2_growth",
    "global_m2_yoy_growth_bitcoin_bitcoin_vs_global_m2_growth",
    "market_cap_bitcoin_bitcoin_dominance",
    "bitcoin_dominance_bitcoin_bitcoin_dominance",
    "us_m2_supply_bitcoin_bitcoin_vs_us_m2_growth",
    "us_m2_yoy_growth_bitcoin_bitcoin_vs_us_m2_growth",
    "whale_index_value_whale_index",
    "long_short_ratio_ls_acc_global",
    "short_percent_ls_acc_global",
    "long_percent_ls_acc_global",
    "taker_buy_volume_usd_z",
    "taker_sell_volume_usd_z",
    "taker_imbalance_z",
    "altcoin_marketcap_index",
    "altcoin_index_index",
    "net_unpnl_bitcoin_bitcoin_net_unrealized_profit_loss",
    "short_term_holder_supply_bitcoin_bitcoin_short_term_holder_supply",
    "ma_350_golden_ratio_multiplier",
    "low_bull_high_2_golden_ratio_multiplier",
    "vocd_bitcoin_bitcoin_reserve_risk",
    "accumulation_high_1_6_golden_ratio_multiplier",
    "ma_110_pi_cycle_indicator",
    "ma_350_mu_2_pi_cycle_indicator",
    "premium_premium_index",
    "premium_rate_premium_index",
    "puell_multiple_puell_multiple",
    "rhodl_ratio_bitcoin_bitcoin_rhodl_ratio",
    "close_basis_futures_basis",
    "hodl_bank_bitcoin_bitcoin_reserve_risk",
    "short_percent_ls_pos_top",
    "close_change_futures_basis",
    "open_basis_futures_basis",
    "long_short_ratio_ls_pos_top",
    "taker_buy_volume_usd",
    "taker_sell_volume_usd",
    "taker_imbalance",
    "long_percent_ls_pos_top",
    "movcd_bitcoin_bitcoin_reserve_risk",
    "long_short_ratio_ls_acc_top",
    "short_percent_ls_acc_top",
    "long_percent_ls_acc_top",
    "open_change_futures_basis",
    "long_term_holder_supply_bitcoin_bitcoin_long_term_holder_supply",
    "reserve_risk_index_bitcoin_bitcoin_reserve_risk",
    "next_halving_stock_flow",
]

# ===================== 时间工具（只统计，不改数据） =====================

def ensure_ms_timestamp(s: pd.Series) -> pd.Series:
    """把任意时间列统一成毫秒 Int64（优先走数值；否则解析成UTC再转毫秒）。"""
    num = pd.to_numeric(s, errors="coerce")
    num_ok = num.notna().mean()
    if num.dropna().size:
        try:
            med_len = num.dropna().astype("int64").astype(str).str.len().median()
        except Exception:
            med_len = 13
        if med_len <= 10:  # 秒 -> 毫秒
            num = num * 1000
    num = num.astype("Int64")

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts_ms = ((dt - epoch) // pd.Timedelta(milliseconds=1))
    ts_ms = ts_ms.astype("Int64")

    return num if num_ok >= ts_ms.notna().mean() else ts_ms


def period_to_ms(v: Optional[str]) -> Optional[int]:
    if v is None or pd.isna(v):
        return None
    x = str(v).strip().lower().replace(" ", "")
    mapping = {
        "1d": {"1d","d","1day","day","24h"},
        "1h": {"1h","h1","60m","60min","60minutes"},
        "4h": {"4h","h4","240m","240min","240minutes"},
    }
    for key, aliases in mapping.items():
        if x in aliases:
            return {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[key]
    m = re.match(r"^(\d+)(ms|s|m|h|d)$", x)
    if m:
        n = int(m.group(1)); u = m.group(2)
        return {"ms":1, "s":1000, "m":60_000, "h":3_600_000, "d":86_400_000}[u] * n
    return None


def infer_step_from_diffs(ts_sorted_unique: pd.Series) -> Optional[int]:
    if ts_sorted_unique.size <= 1:
        return None
    diffs = ts_sorted_unique.diff().dropna().astype("int64")
    if diffs.empty:
        return None
    return int(diffs.value_counts().idxmax())


def list_missing(ts_sorted_unique: pd.Series, step: int, limit_points: int = 2000) -> List[int]:
    missing: List[int] = []
    if ts_sorted_unique.size <= 1:
        return missing
    arr = ts_sorted_unique.to_numpy(dtype="int64")
    for i in range(1, arr.size):
        gap = arr[i] - arr[i-1]
        if gap > step:
            need = min((gap // step) - 1, max(0, limit_points - len(missing)))
            if need > 0:
                start = arr[i-1] + step
                missing.extend([start + k*step for k in range(need)])
            if len(missing) >= limit_points:
                break
    return missing

# ===================== 诊断（新增） =====================

def diagnose_missing_coverage(df: pd.DataFrame, filter_cols: List[str], tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    present_cols = [c for c in filter_cols if c in df.columns]
    if not present_cols:
        return pd.DataFrame(), pd.DataFrame()

    cov = (
        df[present_cols]
        .notna()
        .agg(['count'])
        .T
        .rename(columns={'count': 'non_null'})
    )
    cov['total_rows'] = len(df)
    cov['non_null_ratio'] = cov['non_null'] / cov['total_rows']
    cov = cov.sort_values('non_null_ratio')
    cov.insert(0, 'column', cov.index)

    # 按 period 细分（重点看 1d）
    per = []
    if 'period' in df.columns:
        for p, g in df.groupby('period'):
            sub = g[present_cols].notna().mean().rename('ratio')
            tmp = sub.to_frame().reset_index().rename(columns={'index':'column'})
            tmp['period'] = p
            per.append(tmp)
    per_df = pd.concat(per, ignore_index=True) if per else pd.DataFrame()
    return cov.reset_index(drop=True), per_df

# ===================== 核心流程：删列/删行 + 统计 =====================

def prune_dataframe(
    df: pd.DataFrame,
    how: str,
    *,
    min_k: Optional[int] = None,
    min_ratio: Optional[float] = None,
    filter_periods: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, int, int, int]:
    """
    how:
      - 'any'           : 任一缺失即删（**最苛刻，易误伤**）
      - 'all'           : 仅当这些列全缺失才删（**更安全，推荐**）
      - 'thresh_k'      : 至少有 min_k 个非空；否则删
      - 'thresh_ratio'  : 非空比例 < min_ratio 则删
    可用 --filter-periods 仅对特定 period 应用行过滤（例如只过滤 1h/4h，**不动 1d**）。
    """
    cols_before, rows_before = df.shape[1], df.shape[0]

    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    filter_cols = [c for c in ROW_FILTER_COLS if c in df.columns]
    dropped_rows = 0

    if filter_cols:
        # 仅对指定 period 过滤（默认：全部 period）
        if filter_periods:
            mask_period = df['period'].astype(str).isin(filter_periods) if 'period' in df.columns else pd.Series(False, index=df.index)
        else:
            mask_period = pd.Series(True, index=df.index)

        present_cnt = df[filter_cols].notna().sum(axis=1)
        n = len(filter_cols)

        if how == 'any':
            keep_mask = (present_cnt == n) | (~mask_period)
        elif how == 'all':
            keep_mask = (present_cnt > 0) | (~mask_period)
        elif how == 'thresh_k':
            k = int(min_k or max(1, n//4))
            keep_mask = (present_cnt >= k) | (~mask_period)
        elif how == 'thresh_ratio':
            r = float(min_ratio or 0.5)
            keep_mask = ((present_cnt / n) >= r) | (~mask_period)
        else:
            raise SystemExit(f"[err] unknown --how: {how}")

        before = len(df)
        df = df[keep_mask].copy()
        dropped_rows = before - len(df)

    # 返回：新df、删掉的列数、过滤基列数、删掉的行数
    return df, len(to_drop), len(filter_cols), dropped_rows


def audit_time(df: pd.DataFrame, file_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 找时间列（优先 'timestamp'，否则常见别名）
    if 'timestamp' not in df.columns:
        colmap = {c.lower(): c for c in df.columns}
        aliases = (
            'timestamp','timestamp_ms','ts','ts_ms','unix','unix_time','unix_timestamp',
            'time','date','datetime','day','ds'
        )
        hit = next((colmap[a] for a in aliases if a in colmap), None)
        if hit is None:
            # 没时间列，返回空结果
            return pd.DataFrame(columns=[
                'file','symbol','period','rows','unique_ts','start_dt','end_dt',
                'step_ms','step_ms_inferred','expected_rows','missing_points','gaps','is_continuous','has_duplicates'
            ]), pd.DataFrame(columns=['file','symbol','period','missing_ts','missing_dt'])
        if hit != 'timestamp':
            df = df.rename(columns={hit:'timestamp'})

    df = df.copy()
    df['timestamp'] = ensure_ms_timestamp(df['timestamp'])
    df = df[df['timestamp'].notna()]

    has_symbol = 'symbol' in df.columns
    has_period = 'period' in df.columns
    group_cols: List[str] = []
    if has_symbol: group_cols.append('symbol')
    if has_period: group_cols.append('period')
    if not group_cols:
        df['__dummy__'] = '__all__'
        group_cols = ['__dummy__']

    summaries, gaps_rows = [], []

    for gkey, gdf in df.groupby(group_cols, dropna=False):
        g = gdf.copy()

        # 步长（来自 period 的唯一值）
        step_from_period = None
        if has_period and 'period' in g.columns:
            uniq = g['period'].astype(str).dropna().unique()
            if len(uniq) == 1:
                step_from_period = period_to_ms(uniq[0])

        ts = g['timestamp'].astype('Int64').dropna().astype('int64').drop_duplicates().sort_values()

        rows_total = len(gdf)
        unique_ts = ts.size
        has_dup = rows_total - unique_ts > 0

        if unique_ts == 0:
            summaries.append({
                'file': file_tag,
                'symbol': (gkey[0] if has_symbol and has_period else (gkey if has_symbol else None)) if group_cols[0] != '__dummy__' else None,
                'period': (gkey[1] if has_symbol and has_period else (gkey if (has_period and not has_symbol) else None)) if group_cols[0] != '__dummy__' else None,
                'rows': rows_total, 'unique_ts': unique_ts,
                'start_dt': None, 'end_dt': None,
                'step_ms': step_from_period, 'step_ms_inferred': None,
                'expected_rows': 0, 'missing_points': 0, 'gaps': 0,
                'is_continuous': False, 'has_duplicates': bool(has_dup),
            })
            continue

        start_ms, end_ms = int(ts.iloc[0]), int(ts.iloc[-1])
        start_dt = pd.to_datetime(start_ms, unit='ms', utc=True)
        end_dt   = pd.to_datetime(end_ms, unit='ms', utc=True)

        step_inf = infer_step_from_diffs(ts)
        step_use = step_from_period or step_inf

        if (step_use is None) or (step_use <= 0) or (unique_ts == 1):
            exp_rows = unique_ts
            miss_points = 0
            gaps = 0
            is_cont = (not has_dup)
        else:
            diffs = ts.diff().dropna().astype('int64')
            gaps = int((diffs > step_use).sum())
            exp_rows = int(((end_ms - start_ms) // step_use) + 1)
            miss_points = int(max(0, exp_rows - unique_ts))
            is_cont = (gaps == 0) and (not has_dup)

            # 列出前 2000 个缺口点
            for m in list_missing(ts, step_use, limit_points=2000):
                gaps_rows.append({
                    'file': file_tag,
                    'symbol': (gkey[0] if has_symbol and has_period else (gkey if has_symbol else None)) if group_cols[0] != '__dummy__' else None,
                    'period': (gkey[1] if has_symbol and has_period else (gkey if (has_period and not has_symbol) else None)) if group_cols[0] != '__dummy__' else None,
                    'missing_ts': m,
                    'missing_dt': pd.to_datetime(m, unit='ms', utc=True),
                })

        summaries.append({
            'file': file_tag,
            'symbol': (gkey[0] if has_symbol and has_period else (gkey if has_symbol else None)) if group_cols[0] != '__dummy__' else None,
            'period': (gkey[1] if has_symbol and has_period else (gkey if (has_period and not has_symbol) else None)) if group_cols[0] != '__dummy__' else None,
            'rows': rows_total,
            'unique_ts': unique_ts,
            'start_dt': start_dt, 'end_dt': end_dt,
            'step_ms': step_from_period, 'step_ms_inferred': step_inf,
            'expected_rows': exp_rows, 'missing_points': miss_points, 'gaps': gaps,
            'is_continuous': bool(is_cont), 'has_duplicates': bool(has_dup),
        })

    sum_df = pd.DataFrame(summaries)
    if not sum_df.empty:
        for col in ('start_dt','end_dt'):
            sum_df[col] = pd.to_datetime(sum_df[col], utc=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S%z')

    gaps_df = pd.DataFrame(gaps_rows)
    if not gaps_df.empty:
        gaps_df['missing_dt'] = pd.to_datetime(gaps_df['missing_dt'], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S%z')

    return sum_df, gaps_df

# ===================== IO & 批处理 =====================

def collect_inputs(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.rglob('*.csv'))
    raise FileNotFoundError(p)


def process_one_file(
    in_path: Path,
    out_path: Path,
    how: str,
    *,
    min_k: Optional[int],
    min_ratio: Optional[float],
    filter_periods: Optional[List[str]],
    report_dir: Optional[Path] = None,
    diagnose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(in_path)

    # 诊断缺失覆盖率（基于原始数据，避免“先删后看”）
    if diagnose:
        cov, per = diagnose_missing_coverage(raw, ROW_FILTER_COLS, in_path.name)
        if report_dir is None:
            # 单文件：写到输出旁
            cov_path = out_path.with_suffix('').with_name(out_path.stem + '.missing_coverage.csv')
            per_path = out_path.with_suffix('').with_name(out_path.stem + '.missing_coverage_by_period.csv')
        else:
            cov_path = report_dir / f"{in_path.stem}.missing_coverage.csv"
            per_path = report_dir / f"{in_path.stem}.missing_coverage_by_period.csv"
        if not cov.empty:
            cov.to_csv(cov_path, index=False)
            print(f"[diagnose] {cov_path}")
        if not per.empty:
            per.to_csv(per_path, index=False)
            print(f"[diagnose] {per_path}")

    pruned, n_drop_cols, n_filter_cols, n_drop_rows = prune_dataframe(
        raw, how, min_k=min_k, min_ratio=min_ratio, filter_periods=filter_periods
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pruned.to_csv(out_path, index=False)

    # 统计不同策略的潜在影响，帮助你对比（不改变输出，仅打印）
    if n_filter_cols:
        n = n_filter_cols
        present_cnt = raw[[c for c in ROW_FILTER_COLS if c in raw.columns]].notna().sum(axis=1)
        sim_any = int((present_cnt < n).sum())
        sim_all = int((present_cnt == 0).sum())
        print(
            f"[prune] {in_path.name} -> {out_path.name} | drop_cols={n_drop_cols}, "
            f"filter_cols={n_filter_cols}, drop_rows={n_drop_rows}, shape={raw.shape}->{pruned.shape}"
        )
        print(
            f"[prune.simulate] if how=any -> would drop {sim_any} rows; "
            f"if how=all -> would drop {sim_all} rows"
        )
    else:
        print(
            f"[prune] {in_path.name} -> {out_path.name} | drop_cols={n_drop_cols}, "
            f"filter_cols=0, drop_rows={n_drop_rows}, shape={raw.shape}->{pruned.shape}"
        )

    # 审计基于“删后”的数据
    sum_df, gaps_df = audit_time(pruned, str(out_path))
    return sum_df, gaps_df


def main():
    ap = argparse.ArgumentParser(
        description=(
            "合并流程：删除指定列 & 基于缺失**更安全地**删行；\n"
            "同时输出时间连续性审计 + 缺失覆盖率诊断（避免 1d 被过度抽稀）。"
        )
    )
    ap.add_argument('--input', required=True, help='输入 CSV 文件或目录')
    ap.add_argument('--output', required=False, help='输出 CSV 文件或目录；不指定则生成 *.pruned.csv / <dir>_pruned')
    ap.add_argument('--inplace', action='store_true', help='单文件就地覆盖（与 --output 互斥）')

    # 关键：更安全的行过滤策略
    ap.add_argument('--how', choices=['any','all','thresh_k','thresh_ratio'], default='all',
                    help="行过滤策略：any=任一缺失即删（危险）；all=全缺失才删（默认/安全）；"
                         "thresh_k=少于 K 个非空则删；thresh_ratio=非空比例<r 则删")
    ap.add_argument('--min-k', type=int, default=None, help='--how=thresh_k 时的最小非空列数 K（默认 n//4）')
    ap.add_argument('--min-ratio', type=float, default=None, help='--how=thresh_ratio 时的非空比例 r（默认 0.5）')
    ap.add_argument('--filter-periods', type=str, default=None,
                    help='仅对这些 period 进行行过滤，逗号分隔；例如 "1h,4h"（这样就不动 1d）。')

    # 报表输出目录（目录输入时）
    ap.add_argument('--report-dir', required=False, help='时间审计和缺失诊断输出目录（目录输入时）。')
    ap.add_argument('--no-diagnose', action='store_true', help='关闭缺失覆盖率诊断 CSV 输出')

    args = ap.parse_args()

    in_p = Path(args.input)
    inputs = collect_inputs(in_p)
    if not inputs:
        print('[err] 未找到 CSV'); return

    # 输出路径准备
    if in_p.is_file():
        if args.inplace and args.output:
            raise SystemExit('[err] --inplace 与 --output 互斥（单文件）')
        if args.inplace:
            out_path = in_p
        else:
            out_path = Path(args.output) if args.output else in_p.with_name(in_p.stem + '.pruned.csv')

        filter_periods = [s.strip() for s in args.filter_periods.split(',')] if args.filter_periods else None
        sum_df, gaps_df = process_one_file(
            in_p, out_path, args.how,
            min_k=args.min_k, min_ratio=args.min_ratio,
            filter_periods=filter_periods,
            report_dir=None,
            diagnose=(not args.no_diagnose),
        )

        # 写审计
        rep_sum = out_path.with_suffix('').with_name(out_path.stem + '.time_audit_summary.csv')
        rep_gap = out_path.with_suffix('').with_name(out_path.stem + '.time_audit_gaps_first2000.csv')
        if not sum_df.empty:
            sum_df.to_csv(rep_sum, index=False)
            print(f"[report] {rep_sum}")
        if not gaps_df.empty:
            gaps_df.to_csv(rep_gap, index=False)
            print(f"[report] {rep_gap}")
        return

    # 目录模式：镜像到输出目录；审计统一写到 report-dir
    out_dir = Path(args.output) if args.output else (in_p.parent / (in_p.name + '_pruned'))
    report_dir = Path(args.report_dir) if args.report_dir else (out_dir.parent / (out_dir.name + '_audit'))
    report_dir.mkdir(parents=True, exist_ok=True)

    filter_periods = [s.strip() for s in args.filter_periods.split(',')] if args.filter_periods else None

    all_sum, all_gaps = [], []
    for f in inputs:
        rel = f.relative_to(in_p)
        out_f = out_dir / rel
        out_f.parent.mkdir(parents=True, exist_ok=True)
        sum_df, gaps_df = process_one_file(
            f, out_f, args.how,
            min_k=args.min_k, min_ratio=args.min_ratio,
            filter_periods=filter_periods,
            report_dir=report_dir,
            diagnose=(not args.no_diagnose),
        )
        if not sum_df.empty: all_sum.append(sum_df)
        if not gaps_df.empty: all_gaps.append(gaps_df)

    if all_sum:
        summary = pd.concat(all_sum, ignore_index=True)
        summary.to_csv(report_dir / 'time_audit_summary.csv', index=False)
        print(f"[report] {report_dir / 'time_audit_summary.csv'}")
    if all_gaps:
        gaps = pd.concat(all_gaps, ignore_index=True)
        gaps.to_csv(report_dir / 'time_audit_gaps_first2000.csv', index=False)
        print(f"[report] {report_dir / 'time_audit_gaps_first2000.csv'}")

if __name__ == '__main__':
    main()


#         python  src/prune_and_time_audit.py --input data/merged/full_merged.csv