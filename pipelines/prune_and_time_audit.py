from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Optional, Tuple
import re

# 直接复用 tft 版本逻辑（移植）

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


def ensure_ms_timestamp(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce"); num_ok = num.notna().mean()
    if num.dropna().size:
        try:
            med_len = num.dropna().astype("int64").astype(str).str.len().median()
        except Exception:
            med_len = 13
        if med_len <= 10:
            num = num * 1000
    num = num.astype("Int64")
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts_ms = ((dt - epoch) // pd.Timedelta(milliseconds=1)).astype("Int64")
    return num if num_ok >= ts_ms.notna().mean() else ts_ms


def period_to_ms(v: Optional[str]) -> Optional[int]:
    if v is None or pd.isna(v):
        return None
    x = str(v).strip().lower().replace(" ", "")
    mapping = {"1d": {"1d","d","1day","day","24h"}, "1h": {"1h","h1","60m","60min","60minutes"}, "4h": {"4h","h4","240m","240min","240minutes"}}
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


def list_missing(ts_sorted_unique: pd.Series, step: int, limit_points: int = 2000) -> list[int]:
    missing: list[int] = []
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


def diagnose_missing_coverage(df: pd.DataFrame, filter_cols: List[str], tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    present_cols = [c for c in filter_cols if c in df.columns]
    if not present_cols:
        return pd.DataFrame(), pd.DataFrame()
    cov = (df[present_cols].notna().agg(['count']).T.rename(columns={'count': 'non_null'}))
    cov['total_rows'] = len(df)
    cov['non_null_ratio'] = cov['non_null'] / cov['total_rows']
    cov = cov.sort_values('non_null_ratio')
    cov.insert(0, 'column', cov.index)
    per = []
    if 'period' in df.columns:
        for p, g in df.groupby('period'):
            sub = g[present_cols].notna().mean().rename('ratio')
            tmp = sub.to_frame().reset_index().rename(columns={'index':'column'})
            tmp['period'] = p; per.append(tmp)
    per_df = pd.concat(per, ignore_index=True) if per else pd.DataFrame()
    return cov.reset_index(drop=True), per_df


def prune_dataframe(df: pd.DataFrame, how: str, *, min_k: Optional[int] = None, min_ratio: Optional[float] = None, filter_periods: Optional[List[str]] = None) -> tuple[pd.DataFrame, int, int, int]:
    to_drop = [c for c in DROP_COLS if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
    filter_cols = [c for c in ROW_FILTER_COLS if c in df.columns]
    dropped_rows = 0
    if filter_cols:
        if filter_periods:
            mask_period = df['period'].astype(str).isin(filter_periods) if 'period' in df.columns else pd.Series(False, index=df.index)
        else:
            mask_period = pd.Series(True, index=df.index)
        present_cnt = df[filter_cols].notna().sum(axis=1); n = len(filter_cols)
        if how == 'any':
            keep_mask = (present_cnt == n) | (~mask_period)
        elif how == 'all':
            keep_mask = (present_cnt > 0) | (~mask_period)
        elif how == 'thresh_k':
            k = int(min_k or max(1, n//4)); keep_mask = (present_cnt >= k) | (~mask_period)
        elif how == 'thresh_ratio':
            r = float(min_ratio or 0.5); keep_mask = ((present_cnt / n) >= r) | (~mask_period)
        else:
            raise SystemExit(f"[err] unknown --how: {how}")
        before = len(df); df = df[keep_mask].copy(); dropped_rows = before - len(df)
    return df, len(to_drop), len(filter_cols), dropped_rows


def audit_time(df: pd.DataFrame, file_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'timestamp' not in df.columns:
        colmap = {c.lower(): c for c in df.columns}
        aliases = ('timestamp','timestamp_ms','ts','ts_ms','unix','unix_time','unix_timestamp','time','date','datetime','day','ds')
        hit = next((colmap[a] for a in aliases if a in colmap), None)
        if hit is None:
            return pd.DataFrame(columns=['file','symbol','period','rows','unique_ts','start_dt','end_dt','step_ms','step_ms_inferred','expected_rows','missing_points','gaps','is_continuous','has_duplicates']), pd.DataFrame(columns=['file','symbol','period','missing_ts','missing_dt'])
        if hit != 'timestamp':
            df = df.rename(columns={hit:'timestamp'})
    df = df.copy(); df['timestamp'] = ensure_ms_timestamp(df['timestamp']); df = df[df['timestamp'].notna()]
    has_symbol = 'symbol' in df.columns; has_period = 'period' in df.columns
    group_cols: List[str] = []
    if has_symbol: group_cols.append('symbol')
    if has_period: group_cols.append('period')
    if not group_cols:
        df['__dummy__'] = '__all__'; group_cols = ['__dummy__']
    summaries, gaps_rows = [], []
    for gkey, gdf in df.groupby(group_cols, dropna=False):
        g = gdf.copy(); step_from_period = None
        if has_period and 'period' in g.columns:
            uniq = g['period'].astype(str).dropna().unique()
            if len(uniq) == 1:
                step_from_period = period_to_ms(uniq[0])
        ts = g['timestamp'].astype('Int64').dropna().astype('int64').drop_duplicates().sort_values()
        rows_total = len(gdf); unique_ts = ts.size; has_dup = rows_total - unique_ts > 0
        if unique_ts == 0:
            summaries.append({'file': file_tag,'symbol': (gkey[0] if has_symbol and has_period else (gkey if has_symbol else None)) if group_cols[0] != '__dummy__' else None,'period': (gkey[1] if has_symbol and has_period else (gkey if (has_period and not has_symbol) else None)) if group_cols[0] != '__dummy__' else None,'rows': rows_total, 'unique_ts': unique_ts,'start_dt': None, 'end_dt': None,'step_ms': step_from_period, 'step_ms_inferred': None,'expected_rows': 0, 'missing_points': 0, 'gaps': 0,'is_continuous': False, 'has_duplicates': bool(has_dup)}); continue
        start_ms, end_ms = int(ts.iloc[0]), int(ts.iloc[-1])
        start_dt = pd.to_datetime(start_ms, unit='ms', utc=True); end_dt = pd.to_datetime(end_ms, unit='ms', utc=True)
        step_inf = infer_step_from_diffs(ts); step_use = step_from_period or step_inf
        if (step_use is None) or (step_use <= 0) or (unique_ts == 1):
            exp_rows = unique_ts; miss_points = 0; gaps = 0; is_cont = (not has_dup)
        else:
            diffs = ts.diff().dropna().astype('int64'); gaps = int((diffs > step_use).sum())
            exp_rows = int(((end_ms - start_ms) // step_use) + 1)
            miss_points = int(max(0, exp_rows - unique_ts)); is_cont = (gaps == 0) and (not has_dup)
            for m in list_missing(ts, step_use, limit_points=2000):
                gaps_rows.append({'file': file_tag,'symbol': (gkey[0] if has_symbol and has_period else (gkey if has_symbol else None)) if group_cols[0] != '__dummy__' else None,'period': (gkey[1] if has_symbol and has_period else (gkey if (has_period and not has_symbol) else None)) if group_cols[0] != '__dummy__' else None,'missing_ts': int(m),'missing_dt': pd.to_datetime(int(m), unit='ms', utc=True)})
        summaries.append({'file': file_tag,'symbol': (gkey[0] if has_symbol and has_period else (gkey if has_symbol else None)) if group_cols[0] != '__dummy__' else None,'period': (gkey[1] if has_symbol and has_period else (gkey if (has_period and not has_symbol) else None)) if group_cols[0] != '__dummy__' else None,'rows': rows_total, 'unique_ts': unique_ts,'start_dt': start_dt,'end_dt': end_dt,'step_ms': step_from_period, 'step_ms_inferred': step_inf,'expected_rows': exp_rows, 'missing_points': miss_points, 'gaps': gaps,'is_continuous': bool(is_cont), 'has_duplicates': bool(has_dup)})
    return pd.DataFrame(summaries), pd.DataFrame(gaps_rows)


def collect_inputs(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    return list(p.rglob("*.csv"))


def process_one_file(in_path: Path, out_path: Path, how: str, *, min_k=None, min_ratio=None, filter_periods=None, report_dir: Path | None = None, diagnose: bool = True):
    raw = pd.read_csv(in_path)
    pruned, n_drop_cols, n_filter_cols, n_drop_rows = prune_dataframe(raw, how, min_k=min_k, min_ratio=min_ratio, filter_periods=filter_periods)
    out_path.parent.mkdir(parents=True, exist_ok=True); pruned.to_csv(out_path, index=False)
    if report_dir:
        sum_df, gaps_df = audit_time(pruned, str(out_path))
        if not sum_df.empty:
            sum_df.to_csv(report_dir / (in_path.stem + '.time_audit_summary.csv'), index=False)
        if not gaps_df.empty:
            gaps_df.to_csv(report_dir / (in_path.stem + '.time_audit_gaps_first2000.csv'), index=False)
        if diagnose and n_filter_cols:
            cov = pruned[ROW_FILTER_COLS].notna().mean().rename('ratio').to_frame().reset_index().rename(columns={'index':'column'})
            cov.to_csv(report_dir / (in_path.stem + '.missing_coverage.csv'), index=False)
        print(f"[prune] {in_path.name} -> {out_path.name} | drop_cols={n_drop_cols}, filter_cols={n_filter_cols}, drop_rows={n_drop_rows}, shape={raw.shape}->{pruned.shape}")
    else:
        print(f"[prune] {in_path.name} -> {out_path.name} | drop_cols={n_drop_cols}, filter_cols=0, drop_rows={n_drop_rows}, shape={raw.shape}->{pruned.shape}")
    return audit_time(pruned, str(out_path))


def main():
    ap = argparse.ArgumentParser(description="删除指定列 & 更安全的缺失删行；输出时间连续性审计与覆盖率诊断")
    ap.add_argument('--input', required=True, help='输入 CSV 文件或目录')
    ap.add_argument('--output', required=False, help='输出 CSV 文件或目录；不指定则生成 *.pruned.csv / <dir>_pruned')
    ap.add_argument('--inplace', action='store_true', help='单文件就地覆盖（与 --output 互斥）')
    ap.add_argument('--how', choices=['any','all','thresh_k','thresh_ratio'], default='all')
    ap.add_argument('--min-k', type=int, default=None)
    ap.add_argument('--min-ratio', type=float, default=None)
    ap.add_argument('--filter-periods', type=str, default=None)
    ap.add_argument('--report-dir', required=False)
    ap.add_argument('--no-diagnose', action='store_true')
    args = ap.parse_args()
    in_p = Path(args.input); inputs = collect_inputs(in_p)
    if not inputs:
        print('[err] 未找到 CSV'); return
    if in_p.is_file():
        if args.inplace and args.output:
            raise SystemExit('[err] --inplace 与 --output 互斥（单文件）')
        out_path = in_p if args.inplace else (Path(args.output) if args.output else in_p.with_name(in_p.stem + '.pruned.csv'))
        filter_periods = [s.strip() for s in args.filter_periods.split(',')] if args.filter_periods else None
        sum_df, gaps_df = process_one_file(in_p, out_path, args.how, min_k=args.min_k, min_ratio=args.min_ratio, filter_periods=filter_periods, report_dir=None, diagnose=(not args.no_diagnose))
        rep_sum = out_path.with_suffix('').with_name(out_path.stem + '.time_audit_summary.csv')
        rep_gap = out_path.with_suffix('').with_name(out_path.stem + '.time_audit_gaps_first2000.csv')
        if not sum_df.empty:
            sum_df.to_csv(rep_sum, index=False); print(f"[report] {rep_sum}")
        if not gaps_df.empty:
            gaps_df.to_csv(rep_gap, index=False); print(f"[report] {rep_gap}")
        return
    out_dir = Path(args.output) if args.output else (in_p.parent / (in_p.name + '_pruned'))
    report_dir = Path(args.report_dir) if args.report_dir else (out_dir.parent / (out_dir.name + '_audit'))
    report_dir.mkdir(parents=True, exist_ok=True)
    filter_periods = [s.strip() for s in args.filter_periods.split(',')] if args.filter_periods else None
    all_sum, all_gaps = [], []
    for f in inputs:
        rel = f.relative_to(in_p); out_f = out_dir / rel
        out_f.parent.mkdir(parents=True, exist_ok=True)
        sum_df, gaps_df = process_one_file(f, out_f, args.how, min_k=args.min_k, min_ratio=args.min_ratio, filter_periods=filter_periods, report_dir=report_dir, diagnose=(not args.no_diagnose))
        if not sum_df.empty: all_sum.append(sum_df)
        if not gaps_df.empty: all_gaps.append(gaps_df)
    if all_sum:
        summary = pd.concat(all_sum, ignore_index=True)
        summary.to_csv(report_dir / 'time_audit_summary.csv', index=False); print(f"[report] {report_dir / 'time_audit_summary.csv'}")
    if all_gaps:
        gaps = pd.concat(all_gaps, ignore_index=True)
        gaps.to_csv(report_dir / 'time_audit_gaps_first2000.csv', index=False); print(f"[report] {report_dir / 'time_audit_gaps_first2000.csv'}")

if __name__ == '__main__':
    main()

