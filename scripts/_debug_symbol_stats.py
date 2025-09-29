import os, yaml, re
import pandas as pd

leaf = r'configs/experts/Alpha-Dir-TFT/4h/base'
with open(os.path.join(leaf,'model_config.yaml'),'r',encoding='utf-8') as f:
    mc = yaml.safe_load(f)

p = mc.get('data_path') or 'data/pkl_merged/full_merged.pkl'
if not os.path.exists(p):
    p = 'data/pkl_merged/full_merged.pkl'
print('[data_path]', p, 'exists=', os.path.exists(p))

# read pickle
try:
    df = pd.read_pickle(p)
except Exception as e:
    print('pd.read_pickle failed:', e)
    raise

sym_raw = df['symbol']
ss = sym_raw.astype(str).str.strip()
canon = ss.str.upper().str.replace('/', '_', regex=False).str.replace('-', '_', regex=False)
canon = canon.str.replace(r'^([A-Z]+)(USDT|USD|BUSD|USDC)$', lambda m: f"{m.group(1)}_{m.group(2)}", regex=True)

df2 = df.copy(); df2['symbol'] = canon
vc = df2['symbol'].value_counts(dropna=False)
print('[top symbols]', dict(vc.head(10)))
print('[contains literal "nan" after canon]', int((df2['symbol'].astype(str).str.lower()=='nan').sum()))
sol_mask = df2['symbol'].str.contains('^SOL', regex=True, na=False)
print('[SOL total rows after canon]', int(sol_mask.sum()))
print('[SOL raw variants]', sorted(ss[ss.str.contains('sol', case=False, regex=True, na=False)].unique().tolist())[:10])
print('[rows with slash or dash in raw]', int(ss.str.contains('/|-', regex=True, na=False).sum()))
print('[rows like BTCUSDT pattern in raw]', int(ss.str.fullmatch(r'[A-Za-z]+(USDT|USD|BUSD|USDC)', case=False).sum()))
print('[null symbol rows in raw]', int(sym_raw.isna().sum()))
