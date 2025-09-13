import os
import re
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, List, Dict


def _read_time_range(csv_path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        # read only header to find a timestamp-ish column
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        cand = None
        for k in ['timestamp', 'time', 'ts']:
            if k in cols:
                cand = k; break
        if cand is None:
            # sometimes timestamp exists but case differs
            low = {c.lower(): c for c in cols}
            for k in ['timestamp','time','ts']:
                if k in low:
                    cand = low[k]; break
        if cand is None:
            return None
        df = pd.read_csv(csv_path, usecols=[cand])
        s = pd.to_numeric(df[cand], errors='coerce').dropna()
        if s.empty:
            return None
        # ms vs s
        mx = s.max()
        if mx > 1e12:
            t = pd.to_datetime(s.astype('int64'), unit='ms', utc=True)
        else:
            t = pd.to_datetime(s.astype('int64'), unit='s', utc=True)
        return (t.min(), t.max())
    except Exception:
        return None


def _fmt(ts: Optional[pd.Timestamp]) -> str:
    return ts.tz_convert('UTC').strftime('%Y-%m-%d') if ts is not None else ''


def summarize_onchain(index_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for p in sorted(index_root.rglob('*.csv')):
        rel = p.relative_to(index_root)
        key = str(rel).replace('\\','/')
        rng = _read_time_range(p)
        if rng is None:
            continue
        start, end = rng
        rows.append({'dataset': key, 'start_date': _fmt(start), 'end_date': _fmt(end)})
    return pd.DataFrame(rows).sort_values('dataset')


def _sym_from_pair(fname: str) -> Optional[str]:
    # Extract BTCUSDT or BTC_USDT symbol forms
    m = re.search(r'([A-Z]{2,10})[_]?USDT', fname)
    return m.group(1) if m else None


def summarize_coin_fundamentals(cglass_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    # funding oi/vol
    for kind, sub in [('funding_rate_oi','futures/funding-rate/funding-rate-oi'),
                      ('funding_rate_vol','futures/funding-rate/funding-rate-volume')]:
        d = cglass_root / sub
        if d.exists():
            for p in d.glob('Agg_*_1d.csv'):
                coin = p.stem.split('_')[1] if '_' in p.stem else p.stem
                rng = _read_time_range(p)
                if rng:
                    rows.append({'symbol': f'{coin}_USDT','series': kind,
                                 'start_date': _fmt(rng[0]), 'end_date': _fmt(rng[1])})
    # open interest agg stablecoin
    d = cglass_root / 'futures/open-interest'
    if d.exists():
        for p in d.glob('*_USDT_1d.csv'):
            sym = p.stem.replace('_1d','')
            rng = _read_time_range(p)
            if rng:
                rows.append({'symbol': sym,'series': 'open_interest',
                             'start_date': _fmt(rng[0]), 'end_date': _fmt(rng[1])})
    # long/short ratios
    ls_dirs = [
        ('ls_global', 'futures/long-short/global-long-short-account-ratio'),
        ('ls_top_acc','futures/long-short/top-long-short-account-ratio'),
        ('ls_top_pos','futures/long-short/top-long-short-position-ratio'),
    ]
    for series, sub in ls_dirs:
        d = cglass_root / sub
        if d.exists():
            for p in d.glob('Binance_*_1d.csv'):
                pair = p.stem.replace('Binance_','').replace('_1d','')
                sym = _sym_from_pair(pair)
                if not sym:
                    continue
                rng = _read_time_range(p)
                if rng:
                    rows.append({'symbol': f'{sym}_USDT','series': series,
                                 'start_date': _fmt(rng[0]), 'end_date': _fmt(rng[1])})
    # basis & whale index
    for series, pat in [('basis','futures/futures-basis-Binance-*_1d.csv'),
                        ('whale_index','futures/whale-index-Binance-*_1d.csv')]:
        for p in (cglass_root.glob(pat)):
            pair = p.stem.split('Binance-')[-1].replace('_1d','')
            sym = _sym_from_pair(pair)
            if not sym:
                continue
            rng = _read_time_range(p)
            if rng:
                rows.append({'symbol': f'{sym}_USDT','series': series,
                             'start_date': _fmt(rng[0]), 'end_date': _fmt(rng[1])})
    # taker buy/sell
    d = cglass_root / 'spot/taker-buy-sell-volume'
    if d.exists():
        for p in d.glob('*_1d.csv'):
            pair = p.stem.replace('_1d','')
            sym = _sym_from_pair(pair)
            if not sym:
                continue
            rng = _read_time_range(p)
            if rng:
                rows.append({'symbol': f'{sym}_USDT','series': 'taker_buy_sell',
                             'start_date': _fmt(rng[0]), 'end_date': _fmt(rng[1])})
    # borrow interest (per coin)
    d = cglass_root / 'index'
    for p in d.glob('borrow_interest_rate-Binance-*-1d.csv'):
        parts = p.stem.split('-')
        coin = parts[2] if len(parts) >= 4 else _sym_from_pair(p.stem) or 'UNKNOWN'
        rng = _read_time_range(p)
        if rng:
            rows.append({'symbol': f'{coin}_USDT','series': 'borrow_interest_rate',
                         'start_date': _fmt(rng[0]), 'end_date': _fmt(rng[1])})
    return pd.DataFrame(rows).sort_values(['symbol','series'])


def main():
    root = Path('data/cglass')
    out_dir = Path('reports/time_ranges'); out_dir.mkdir(parents=True, exist_ok=True)
    # On-chain/macros (index)
    onchain = summarize_onchain(root / 'index')
    on_path = out_dir / 'onchain_time_ranges.csv'
    onchain.to_csv(on_path, index=False)
    print(f'✅ on-chain summary -> {on_path}')
    # Coin fundamentals (futures/spot per symbol)
    fund = summarize_coin_fundamentals(root)
    fu_path = out_dir / 'coin_fundamentals_time_ranges.csv'
    fund.to_csv(fu_path, index=False)
    print(f'✅ coin fundamentals summary -> {fu_path}')

    # Print head previews
    print('\n[On-chain top 10]')
    print(onchain.head(10).to_string(index=False))
    print('\n[Coin fundamentals top 10]')
    print(fund.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
