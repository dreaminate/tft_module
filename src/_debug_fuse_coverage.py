import os
import pandas as pd

def main():
    base = 'data/merged/full_merged_with_fundamentals_expert_group.csv'
    if not os.path.exists(base):
        base = 'data/merged/full_merged_with_fundamentals.csv'
    cols_file = 'data/merged/fundamental_columns.txt'
    with open(cols_file, 'r', encoding='utf-8') as f:
        new_cols = [x.strip() for x in f if x.strip()]
    usecols = ['timestamp','symbol','period'] + new_cols
    print(f'Reading: {base}')
    df = pd.read_csv(base, usecols=[c for c in usecols if c in pd.read_csv(base, nrows=0).columns], low_memory=False)
    print('Rows:', len(df), 'Cols:', len(df.columns))
    present = [c for c in new_cols if c in df.columns]
    cov = df.groupby('period')[present].apply(lambda g: g.notna().mean())
    cov.T.to_csv('data/merged/fuse_audit/quick_by_period_coverage.csv')
    print('Saved by-period coverage -> data/merged/fuse_audit/quick_by_period_coverage.csv')
    # earliest ts per column
    rows = []
    for c in present:
        idx = df[c].notna()
        ts = int(df.loc[idx, 'timestamp'].min()) if idx.any() else None
        rows.append({'column': c, 'first_ts': ts})
    pd.DataFrame(rows).to_csv('data/merged/fuse_audit/first_timestamp_per_column.csv', index=False)
    print('Saved first ts per column -> data/merged/fuse_audit/first_timestamp_per_column.csv')

if __name__ == '__main__':
    main()
